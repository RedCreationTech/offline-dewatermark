"""WeVisOne 提示词分割推理服务（stdio JSON-lines 协议）。

用 WeVisOne 自带 venv 运行（主环境没有 torch）：
    out/third_party/WeVisOne-VL/.venv/Scripts/python.exe tools/wevisone_server.py

协议：每行一个 JSON 对象。
启动后先输出一行 {"status": "ready", ...} 或 {"status": "failed", "error": ...}（随后退出）。
请求：{"cmd": "ping"} 或 {"cmd": "segment", "image_png_b64": "...", "query": "..."}
响应：{"ok": true, ...} 或 {"ok": false, "error": "..."}
segment 响应实例框为原图像素坐标 {"instances": [{"box": [x0, y0, x1, y1]}],
"latency_ms": n, "model": str}。matte 不回传（GUI 端不使用，见设计文档 Q3）。

--stub 模式不加载模型，返回固定中央框，用于协议联调与无 GPU 测试。
环境变量 WM_WEVISONE_DEC_BATCH 控制高分辨率抠图精修的单批实例数（8GB 卡需调小）。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

MAX_LINE = 64 * 1024 * 1024  # base64 大图会撑长行，给足上限并拒绝失控输入


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _recv() -> dict | None:
    line = sys.stdin.buffer.readline(MAX_LINE + 1)
    if not line:
        return None
    if len(line) > MAX_LINE:
        raise ValueError("请求行超长")
    return json.loads(line.decode("utf-8"))


def _load_model(model_path: str, device: str):
    import torch
    from wevisone_vl import WeVisOneVL

    dtype = "float32"
    if device == "cuda:0":
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    decoder_batch = int(os.environ.get("WM_WEVISONE_DEC_BATCH", "16"))
    return WeVisOneVL(model_path, backend="transformers", device=device,
                      dtype=dtype, decoder_batch_size=decoder_batch), dtype


def _stub_segment() -> dict:
    return {"ok": True, "instances": [{"box": [64, 48, 192, 144]}],
            "latency_ms": 1, "model": "stub"}


def _segment(model, image_png_b64: str, query: str) -> dict:
    import numpy as np
    from PIL import Image

    raw = base64.b64decode(image_png_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = image.size
    started = time.perf_counter()
    result = model.predict(image, task="segmentation", query=query)
    latency_ms = int((time.perf_counter() - started) * 1000)
    instances = []
    for box in result.boxes:
        x0, y0, x1, y1 = box
        px0, py0 = int(round(x0 * w)), int(round(y0 * h))
        px1, py1 = int(round(x1 * w)), int(round(y1 * h))
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(w, px1), min(h, py1)
        if px1 - px0 >= 4 and py1 - py0 >= 4:
            instances.append({"box": [px0, py0, px1, py1]})
    return {"ok": True, "instances": instances, "latency_ms": latency_ms,
            "model": getattr(model, "model_name_or_path", "wevisone")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/WeVisOne-VL-2B")
    parser.add_argument("--device", default=os.environ.get("WM_WEVISONE_DEVICE", "cpu"))
    parser.add_argument("--stub", action="store_true", help="不加载模型，返回固定框")
    args = parser.parse_args()

    if args.stub:
        _send({"status": "ready", "model": "stub", "device": "stub"})
    else:
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            model, dtype = _load_model(args.model, args.device)
        except Exception as exc:  # 启动失败要让客户端立刻知道原因
            _send({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            return 1
        _send({"status": "ready", "model": args.model, "device": args.device,
               "dtype": dtype})

    while True:
        try:
            request = _recv()
        except Exception as exc:
            _send({"ok": False, "error": f"协议错误: {exc}"})
            continue
        if request is None:  # 管道关闭，退出
            return 0
        cmd = request.get("cmd")
        try:
            if cmd == "ping":
                _send({"ok": True, "pong": True})
            elif cmd == "segment":
                if args.stub:
                    _send(_stub_segment())
                else:
                    _send(_segment(model, request["image_png_b64"],
                                   str(request.get("query", ""))))
            else:
                _send({"ok": False, "error": f"未知命令: {cmd}"})
        except Exception as exc:
            _send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    sys.exit(main())
