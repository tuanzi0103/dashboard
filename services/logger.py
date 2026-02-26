import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 统一日志目录：放在项目根目录下的 logs/app.log
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "app.log"

# 非 Streamlit 场景兜底（比如 CLI / 单元测试）
_LOG_INITIALIZED = False

def init_logging():
    """Initialize file logging once per process (and once per Streamlit session if available)."""
    global _LOG_INITIALIZED

    # Streamlit 场景：用 session_state 防止 rerun 重复加 handler
    try:
        import streamlit as st
        if st.session_state.get("_log_initialized"):
            return
    except Exception:
        st = None

    if _LOG_INITIALIZED:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("manlyfarm")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # ✅ 避免重复输出到 root logger

    # ✅ 防止重复 handler（无论 Streamlit rerun 还是多次 import）
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        handler = RotatingFileHandler(
            LOG_FILE, maxBytes=2_000_000, backupCount=2, encoding="utf-8"
        )
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    _LOG_INITIALIZED = True

    try:
        if st is not None:
            st.session_state["_log_initialized"] = True
    except Exception:
        pass


def log_info(msg: str):
    logging.getLogger("manlyfarm").info(msg)

def log_warning(msg: str):
    logging.getLogger("manlyfarm").warning(msg)

def log_error(msg: str):
    logging.getLogger("manlyfarm").error(msg)
