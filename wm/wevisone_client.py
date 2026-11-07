"""WeVisOne 推理服务的 GUI 端客户端：子进程生命周期 + stdio JSON-lines 协议。

服务脚本在 tools/wevisone_server.py，必须用 WeVisOne venv 解释器运行
（主 venv 没有 torch）。常驻/按需两种模式由调用方决定何时 start()/stop()：
常驻模式随「提示词定位」方式选中而预热、切走即停；按需模式每轮处理启动一次、
跑完即停（见 docs/提示词水印定位辅助功能设计_20261107.md 3.1）。

线程模型：GUI 线程负责 start/stop，BatchWorker 线程并发调用 segment()；
内部用锁串行化管道读写。进程意外退出时请求抛 ServiceDied，由调用方重启一次。
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_SCRIPT = ROOT / "tools" / "wevisone_server.py"
VENV_PYTHON = ROOT / "out" / "third_party" / "WeVisOne-VL" / ".venv" / "Scripts" / "python.exe"
LOG_FILE = ROOT / "out" / "wevisone_service.log"
DEFAULT_MODEL = os.environ.get("WM_WEVISONE_MODEL", "models/WeVisOne-VL-2B")
SEGMENT_TIMEOUT_S = 120  # 设计 Q11：单页调用超时
START_TIMEOUT_S = 300    # 冷启动含模型加载


class ServiceError(RuntimeError):
    pass


class ServiceDied(ServiceError):
    """进程在请求途中退出（常驻模式崩溃），调用方可重启一次。"""


class _LineReader(threading.Thread):
    """按行读管道，避免 Windows 上 select 不可用导致的阻塞问题。"""

    def __init__(self, stream) -> None:
        super().__init__(daemon=True)
        self._stream = stream
        self.queue: list[bytes] = []
        self.closed = False
        self._event = threading.Event()

    def run(self) -> None:
        while True:
            line = self._stream.readline()
            if not line:
                break
            self.queue.append(line)
            self._event.set()
        self.closed = True
        self._event.set()

    def pop(self, timeout: float) -> bytes | None:
        end = time.monotonic() + timeout
        while True:
            if self.queue:
                return self.queue.pop(0)
            if self.closed:
                return None
            remaining = end - time.monotonic()
            if remaining <= 0:
                return None
            self._event.wait(remaining)
            self._event.clear()


class Service:
    def __init__(self, *, executable: str | Path | None = None,
                 model: str | None = None, log_file: Path | None = None,
                 stub: bool = False) -> None:
        self.executable = str(executable) if executable else str(VENV_PYTHON)
        self.model = model or DEFAULT_MODEL
        self.log_file = Path(log_file) if log_file else LOG_FILE
        self.stub = stub
        self._proc: subprocess.Popen | None = None
        self._reader: _LineReader | None = None
        self._lock = threading.Lock()
        self.last_latency_ms: int | None = None
        self.last_model: str | None = None

    # ---- 生命周期 -------------------------------------------------------

    @staticmethod
    def available() -> bool:
        return VENV_PYTHON.is_file()

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, timeout: float = START_TIMEOUT_S) -> dict:
        """启动服务并等待就绪行；已运行则直接返回。抛 ServiceError。"""
        with self._lock:
            if self.running:
                return {"status": "ready"}
            if not Path(self.executable).is_file():
                raise ServiceError("WeVisOne 环境未安装，无法使用提示词定位。")
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            log = open(self.log_file, "ab", buffering=0)
            self._log = log
            env = dict(os.environ)
            env.setdefault("HF_HUB_OFFLINE", "1")
            env.setdefault("PYTHONUTF8", "1")
            env.setdefault("WM_WEVISONE_DEVICE", "cuda:0")
            try:
                command = [self.executable, str(SERVER_SCRIPT), "--model", self.model]
                if self.stub:
                    command.append("--stub")
                self._proc = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                    cwd=str(ROOT), env=env)
            except OSError as exc:
                log.close()
                raise ServiceError(f"无法启动提示词模型服务: {exc}") from exc
            self._reader = _LineReader(self._proc.stdout)
            self._reader.start()
            line = self._reader.pop(timeout)
            if line is None:
                self._kill_locked()
                raise ServiceError("提示词模型服务启动超时。")
            try:
                ready = json.loads(line.decode("utf-8"))
            except ValueError:
                self._kill_locked()
                raise ServiceError("提示词模型服务就绪信号无效。")
            if ready.get("status") != "ready":
                self._kill_locked()
                raise ServiceError(f"提示词模型服务启动失败: {ready.get('error', '未知原因')}")
            self.last_model = ready.get("model")
            return ready

    def stop(self) -> None:
        with self._lock:
            self._kill_locked()

    def _kill_locked(self) -> None:
        proc, self._proc = self._proc, None
        self._reader = None
        log, self._log = getattr(self, "_log", None), None
        if log is not None:
            try:
                log.close()
            except Exception:
                pass
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # ---- 请求 -----------------------------------------------------------

    def ping(self, timeout: float = 5.0) -> bool:
        try:
            return bool(self._request({"cmd": "ping"}, timeout).get("ok"))
        except ServiceError:
            return False

    def segment(self, image_rgb, query: str,
                timeout: float = SEGMENT_TIMEOUT_S) -> list[tuple[int, int, int, int]]:
        """对整图跑提示词分割，返回实例框列表（原图像素坐标 xyxy）。"""
        import numpy as np
        from PIL import Image

        array = np.ascontiguousarray(image_rgb[..., :3])
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, "PNG")
        payload = {
            "cmd": "segment",
            "image_png_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "query": query,
        }
        response = self._request(payload, timeout)
        boxes = []
        for item in response.get("instances", []):
            box = item.get("box")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                boxes.append(tuple(int(v) for v in box))
        self.last_latency_ms = response.get("latency_ms")
        self.last_model = response.get("model", self.last_model)
        return boxes

    def _request(self, payload: dict, timeout: float) -> dict:
        with self._lock:
            if not self.running:
                raise ServiceError("提示词模型服务未启动。")
            body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            try:
                self._proc.stdin.write(body)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._kill_locked()
                raise ServiceDied("提示词模型服务已退出。") from exc
            line = self._reader.pop(timeout)
            if line is None:
                died = self._reader.closed
                self._kill_locked()
                if died:
                    raise ServiceDied("提示词模型服务已退出。")
                raise ServiceError("提示词模型服务响应超时。")
            response = json.loads(line.decode("utf-8"))
        if not response.get("ok"):
            raise ServiceError(str(response.get("error", "未知错误")))
        return response


def default_service() -> Service:
    return Service()
