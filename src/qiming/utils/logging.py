# -*- coding: utf-8 -*-
"""日志配置。

统一为整个后端配置日志：
    - 控制台输出（跟随终端）；
    - 滚动文件输出（logs/app.log，单文件 5MB，保留 5 份备份），落盘的日志
      便于服务异常后回溯排查；
    - 级别可通过环境变量 LOG_LEVEL 调整（DEBUG / INFO / WARNING / ERROR）。

configure_logging 可重复调用（重复调用只调整级别，不会重复挂 handler）。
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s (%(module)s:%(lineno)d): %(message)s"

# 项目根目录下的 logs/（.gitignore 已忽略）
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[3] / "logs"
LOG_FILE_NAME = "app.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5


def configure_logging(level: str | None = None, log_dir: str | Path | None = None) -> None:
    """初始化根日志：控制台 + 滚动文件。

    参数:
        level:   日志级别字符串；缺省读环境变量 LOG_LEVEL，再缺省 INFO。
        log_dir: 日志目录；缺省读环境变量 LOG_DIR，再缺省 <项目根>/logs。
                 传入空串可禁用文件输出（部分测试场景使用）。
    """
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level_value = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()

    # 已初始化过：只调整级别，避免 pytest / uvicorn --reload 下重复挂 handler
    if getattr(root, "_qiming_configured", False):
        root.setLevel(level_value)
        return

    root.setLevel(level_value)
    formatter = logging.Formatter(LOG_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    dir_path = log_dir or os.getenv("LOG_DIR") or DEFAULT_LOG_DIR
    if str(dir_path).strip():
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                Path(dir_path) / LOG_FILE_NAME,
                maxBytes=LOG_MAX_BYTES,
                backupCount=LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:  # 目录不可写时降级为仅控制台，不让服务起不来
            root.warning("日志文件初始化失败，仅输出到控制台: %s", exc)

    # 降低第三方库的噪音
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "httpx", "websockets"):
        logging.getLogger(noisy).setLevel(max(level_value, logging.WARNING))

    root._qiming_configured = True  # type: ignore[attr-defined]
