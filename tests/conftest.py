# -*- coding: utf-8 -*-
"""pytest 全局配置：测试用独立的临时 SQLite，避免污染开发数据库。"""

import os
import tempfile

# 必须在导入 src.api.app 之前设置好环境
os.environ.setdefault("DATABASE_TYPE", "sqlite")
os.environ.setdefault("SQLITE_PATH", os.path.join(tempfile.mkdtemp(prefix="qiming_test_"), "test.db"))
# 测试强制走 mock 调度器：真实模型的意图/回复存在不确定性，且会产生调用费用
os.environ.setdefault("LLM_MODE", "mock")

# 保证仓库根目录在 sys.path 上（pytest 从任意目录启动时也能 import src.*）
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
