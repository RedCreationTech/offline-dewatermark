"""运行日志：同时写文件和控制台，级别可配置、可在程序运行中调整。

级别来源（优先级从高到低）：
- 环境变量 WM_LOG_LEVEL（如 set WM_LOG_LEVEL=DEBUG）；
- config/logging.json 的 "level" 字段；
- 默认 INFO。

GUI 启动后会监听 config/logging.json 的变化，改文件即生效；
界面里的「日志级别」下拉会立即生效并写回配置文件。
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
from pathlib import Path

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
DEFAULT_LEVEL = "INFO"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "logging.json"
LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FILE = LOG_DIR / "wm.log"

_logger = logging.getLogger("wm")


def configured_level() -> str:
    """读取生效的日志级别（环境变量 > config/logging.json > 默认）。"""
    env = os.environ.get("WM_LOG_LEVEL", "").strip().upper()
    if env in LEVELS:
        return env
    try:
        level = str(json.loads(CONFIG_PATH.read_text("utf-8")).get("level", "")).upper()
        if level in LEVELS:
            return level
    except Exception:
        pass
    return DEFAULT_LEVEL


def apply_level(level: str) -> str:
    """调整 wm 记录器及所有处理器的级别，立即生效；返回生效级别名。"""
    level = level.upper()
    if level not in LEVELS:
        level = DEFAULT_LEVEL
    _logger.setLevel(level)
    for handler in _logger.handlers:
        handler.setLevel(level)
    _logger.info("日志级别已调整为 %s", level)
    return level


def persist_level(level: str) -> None:
    """把级别写回 config/logging.json（供下次启动和文件监听使用）。"""
    level = level.upper()
    if level not in LEVELS:
        return
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps({"level": level}, ensure_ascii=False, indent=2) + "\n", "utf-8")
    except OSError:
        _logger.warning("无法写入日志配置 %s", CONFIG_PATH, exc_info=True)


def setup_logging(level: str | None = None) -> Path:
    """初始化日志（幂等）：logs/wm.log 滚动备份，同时输出到控制台。"""
    if _logger.handlers:
        apply_level(level or configured_level())
        return LOG_FILE
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    level = level or configured_level()
    _logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    _logger.addHandler(file_handler)
    _logger.addHandler(console)
    _logger.info("日志启动：级别 %s，文件 %s", level, LOG_FILE)
    return LOG_FILE
