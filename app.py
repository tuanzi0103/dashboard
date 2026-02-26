import streamlit as st
from datetime import datetime, timedelta, date
import pandas as pd

from services.logger import init_logging, LOG_FILE, log_info, log_warning, log_error

st.set_page_config(
    page_title="Vie Manly Analytics",
    layout="wide",
    initial_sidebar_state="auto"
)
init_logging()


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from services.db import get_db_path
import os
import pandas as pd
from services.analytics import load_all, rebuild_high_level_summary, rebuild_inventory_summary
from services.db import get_db, init_database
from services.ingestion import ingest_excel, ingest_csv, init_db_from_drive_once
from charts.high_level import show_high_level
from charts.sales_report import show_sales_report
from charts.inventory import show_inventory
from charts.product_mix_only import show_product_mix_only
from charts.customer_segmentation import show_customer_segmentation
from init_db import init_db
import subprocess
import sys
from services.ingestion import ingest_from_drive_all
import platform
import numpy as np
from datetime import datetime, timedelta

import psutil

init_database()

def check_memory():
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    usage_ratio = used_gb / total_gb

    if usage_ratio > 0.85:
        st.warning(f"⚠️ Memory usage high ({usage_ratio*100:.1f}%). Please refresh occasionally.")



# 关闭文件监控，避免 Streamlit Cloud 报 inotify 错误
os.environ["WATCHDOG_DISABLE_FILE_WATCH"] = "true"

# ✅ 确保 SQLite 文件和表结构存在
init_db()  # 必须先初始化数据库表结构

if "drive_initialized" not in st.session_state:
    ok = init_db_from_drive_once()
    if ok:
        rebuild_high_level_summary()
        rebuild_inventory_summary()
        st.session_state.drive_initialized = True
    else:
        # ingest 被锁 / Drive 未 ready，不要标记已初始化
        st.warning("⏳ Database is initializing. Please wait...")
        st.stop()



st.markdown("<h1 style='font-size:26px; font-weight:700;'>📊 Vie Manly Dashboard</h1>", unsafe_allow_html=True)

@st.cache_data(show_spinner="loading...")
def load_db_cached(db_mtime):
    db = get_db()
    return load_all(db=db)

def reload_db_cache():
    st.session_state.pop("db_cache", None)
    db_path = get_db_path()
    db_mtime = os.path.getmtime(db_path)
    st.session_state.db_cache = load_db_cached(db_mtime)

BAD_DATES = {
    date(2025, 8, 18),
    date(2025, 8, 19),
    date(2025, 8, 20),
}
def check_missing_data(tx, inv):
    """
    分开检查交易和库存的缺失日期：

    - 交易（transactions）：
        * 从固定的起始日期 tx_start_date 开始（你可以根据需要改）
        * 到今天为止，每一天如果在数据库里完全没有交易记录，就标记为缺失

    - 库存（inventory）：
        * 从固定的起始日期 inv_start_date 开始（你明确说要从 2025-11-01）
        * 到今天为止，每一天如果在数据库里没有任何 inventory 记录，就标记为缺失
    """
    missing_info = {
        "transaction_dates": [],
        "inventory_dates": [],
    }

    today = datetime.now().date()

    # ===== 1. 交易缺失检查 =====
    # 如果你以后想改成从 2024-01-01 开始检测，可以把下面这行改成 date(2024, 1, 1)
    tx_start_date = date(2024, 1, 1)

    if tx is not None and not tx.empty and "Datetime" in tx.columns:
        # 把 Datetime 列安全地转成日期
        tx_dates_series = pd.to_datetime(tx["Datetime"], errors="coerce").dt.date
        tx_dates = set(d for d in tx_dates_series.dropna())

        # 只在 tx_start_date ~ today 这个区间内检查
        if tx_start_date <= today:
            all_days = [
                tx_start_date + timedelta(days=i)
                for i in range((today - tx_start_date).days + 1)
            ]
            missing_tx = [
                d for d in all_days
                if d not in tx_dates and d not in BAD_DATES
            ]

            missing_info["transaction_dates"] = missing_tx

    # ===== 2. 库存缺失检查 =====
    # 按你的要求：inventory 固定从 2025-11-01 往后检查
    inv_start_date = date(2025, 11, 1)

    if inv is not None and not inv.empty and "source_date" in inv.columns:
        inv_dates_series = pd.to_datetime(inv["source_date"], errors="coerce").dt.date
        inv_dates = set(d for d in inv_dates_series.dropna())

        if inv_start_date <= today:
            all_days = [
                inv_start_date + timedelta(days=i)
                for i in range((today - inv_start_date).days + 1)
            ]
            missing_inv = [d for d in all_days if d not in inv_dates]
            missing_info["inventory_dates"] = missing_inv

    return missing_info

if "db_auto_reloaded" not in st.session_state:
    reload_db_cache()
    st.session_state.db_auto_reloaded = True

tx, mem, inv = st.session_state.db_cache


# === Sidebar ===
st.sidebar.header("⚙️ Settings")

# === 数据缺失预警 ===
missing_data = check_missing_data(tx, inv)

if missing_data['transaction_dates'] or missing_data['inventory_dates']:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚠️ Data missing warning")

    if missing_data['transaction_dates']:
        st.sidebar.error("**Missing transaction date:**")
        # 显示最近7天的缺失日期，其他的折叠显示
        recent_missing = sorted(missing_data['transaction_dates'])[-7:]
        for date in recent_missing:
            st.sidebar.write(f"📅 {date.strftime('%Y-%m-%d')}")

        if len(missing_data['transaction_dates']) > 7:
            with st.sidebar.expander(f"check all {len(missing_data['transaction_dates'])} missing dates"):
                for date in sorted(missing_data['transaction_dates']):
                    st.write(f"📅 {date.strftime('%Y-%m-%d')}")

    if missing_data['inventory_dates']:
        st.sidebar.warning("**Missing inventory date:**")
        # 显示最近7天的缺失日期，其他的折叠显示
        recent_missing = sorted(missing_data['inventory_dates'])[-7:]
        for date in recent_missing:
            st.sidebar.write(f"📦 {date.strftime('%Y-%m-%d')}")

        if len(missing_data['inventory_dates']) > 7:
            with st.sidebar.expander(f"check all {len(missing_data['inventory_dates'])} missing dates"):
                for date in sorted(missing_data['inventory_dates']):
                    st.write(f"📦 {date.strftime('%Y-%m-%d')}")

# 文件上传 - 添加上传状态跟踪
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = set()

uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

# ✅ 改造：上传只“暂存”，点击按钮才 ingest（只处理新文件，不重建 DB）
if "pending_uploads" not in st.session_state:
    st.session_state.pending_uploads = {}  # {filename: UploadedFile}

# 收集本次选择的“新文件”（不立刻 ingest）
if uploaded_files:
    for f in uploaded_files:
        # 这个 session 里已经 ingest 过的，直接跳过
        if f.name in st.session_state.uploaded_file_names:
            continue
        # 暂存等待按钮触发 ingest
        st.session_state.pending_uploads[f.name] = f

# 给用户提示
if st.session_state.pending_uploads:
    st.sidebar.info(
        f"📥 {len(st.session_state.pending_uploads)} new file(s) ready. "
        f"Click '🔄 Refresh New Files' to ingest."
    )

if st.sidebar.button("🔄 Refresh New Files"):
    from services.ingestion import ingest_csv, ingest_excel, ingest_file_lock
    from services.analytics import update_high_level_summary_by_db_diff, rebuild_inventory_summary

    ingested_any = False

    with ingest_file_lock() as locked:
        if not locked:
            st.warning("⏳ Another ingestion is running. Please try again.")
            st.stop()

        # === 1️⃣ Ingest 新文件 ===
        pending = list(st.session_state.get("pending_uploads", {}).items())
        for filename, uploaded_file in pending:
            try:
                if filename.lower().endswith(".csv"):
                    ingest_csv(uploaded_file, filename)
                elif filename.lower().endswith(".xlsx"):
                    ingest_excel(uploaded_file, filename)
                st.session_state.uploaded_file_names.add(filename)
                ingested_any = True
            except Exception as e:
                st.error(f"❌ Failed to ingest {filename}: {e}")

        st.session_state.pending_uploads.clear()

    # === 2️⃣ 🔥 关键：先重算 Summary，再刷新缓存 ===
    # 必须在 reload_db_cache 之前执行，确保数据库里的 Summary 表是最新的
    update_high_level_summary_by_db_diff()
    rebuild_inventory_summary()

    # === 3️⃣ 🔥 关键：清除 Streamlit 的所有数据缓存 ===
    # 这会强制 load_summary() 等函数重新从数据库读取最新的 Summary 表
    st.cache_data.clear()

    # === 4️⃣ 刷新原始数据的 cache (tx, mem, inv) ===
    reload_db_cache()

    if ingested_any:
        st.success("✅ New files ingested and data refreshed")
    else:
        st.info("ℹ️ Data reloaded")

    st.rerun()

# ===============================
# 🛠️ Database maintenance
# ===============================
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Database")

if st.sidebar.button("Clear & Rebuild Database"):
    # 1️⃣ 立即清除 Streamlit 所有全局函数缓存 (核心步骤)
    # 这确保了 charts 里的 load_summary 等函数下次运行时必读数据库，而不是读内存
    st.cache_data.clear()

    # 2️⃣ 清理 Session 中的旧引用
    if "db_cache" in st.session_state:
        del st.session_state["db_cache"]

    # 3️⃣ 执行从 Drive 的全量重新同步
    # 注意：确保你的 ingest_from_drive_all 内部会先 DROP/TRUNCATE 掉原始表
    ok = ingest_from_drive_all()
    if not ok:
        st.sidebar.warning("⏳ Database is busy. Please try again.")
        st.stop()

    # 4️⃣ 强制初始化表结构
    init_database()

    # 5️⃣ 物理清空 Summary 表，确保没有残留
    with get_db() as conn:
        conn.execute("DELETE FROM high_level_daily")
        conn.execute("DELETE FROM inventory_summary")
        conn.commit()

    # 6️⃣ 等待数据写入物理磁盘的缓冲（针对某些环境下的 SQLite 延迟）
    import time

    time.sleep(1)

    # 7️⃣ 执行重算
    rebuild_high_level_summary()
    rebuild_inventory_summary()

    # 8️⃣ 重新加载缓存到 session_state
    reload_db_cache()

    st.sidebar.success("✅ Database fully rebuilt and cache cleared.")
    st.rerun()

# --- 2) Refresh (cache only) ---
if st.sidebar.button("🔄 Refresh data"):
    reload_db_cache()
    st.sidebar.success("Reloading data…")
    st.rerun()


# --- 3) Debug Snapshot ---
if st.sidebar.button("Debug Snapshot"):
    try:
        conn = get_db()

        row = conn.execute("PRAGMA database_list").fetchone()
        db_path = row[2] if row and len(row) >= 3 else None

        log_info("🧪 DEBUG SNAPSHOT")
        log_info(f"🗄️ DB path: {db_path}")

        tx_stats = conn.execute("""
            SELECT 
                MIN(date(Datetime)),
                MAX(date(Datetime)),
                COUNT(*),
                COUNT(DISTINCT date(Datetime))
            FROM transactions
        """).fetchone()

        log_info(
            f"📊 transactions: min_date={tx_stats[0]}, "
            f"max_date={tx_stats[1]}, rows={tx_stats[2]}, "
            f"distinct_days={tx_stats[3]}"
        )

        inv_stats = conn.execute("""
            SELECT 
                MIN(source_date),
                MAX(source_date),
                COUNT(*),
                COUNT(DISTINCT source_date)
            FROM inventory
        """).fetchone()

        log_info(
            f"📦 inventory: min_date={inv_stats[0]}, "
            f"max_date={inv_stats[1]}, rows={inv_stats[2]}, "
            f"distinct_days={inv_stats[3]}"
        )

        st.sidebar.success("Debug snapshot written to log.")

    except Exception as e:
        log_error(f"❌ DEBUG SNAPSHOT failed: {e}")
        st.sidebar.error("Debug snapshot failed. Check logs.")


with st.sidebar.expander("🪵 Logs"):
    st.caption(f"Log file: {LOG_FILE}")
    try:
        log_text = LOG_FILE.read_text(encoding="utf-8")
    except Exception:
        log_text = ""
    tail = "\n".join(log_text.splitlines()[-60:])
    st.text_area("Latest log lines", tail, height=220)
    st.download_button("Download app.log", log_text, file_name="app.log", mime="text/plain")


# === 单位选择 ===
st.sidebar.subheader("📏 Units")

if inv is not None and not inv.empty and "Unit" in inv.columns:
    units_available = sorted(inv["Unit"].dropna().unique().tolist())
else:
    units_available = ["Gram 1.000", "Kilogram 1.000", "Milligram 1.000"]

conn = get_db()
try:
    rows = conn.execute("SELECT name FROM units").fetchall()
    db_units = [r[0] for r in rows]  # 修复这里的索引错误
except Exception:
    db_units = []

all_units = sorted(list(set(units_available + db_units)))
unit = st.sidebar.selectbox("Choose unit", all_units)

new_unit = st.sidebar.text_input("Add new unit")
if st.sidebar.button("➕ Add Unit"):
    if new_unit and new_unit not in all_units:
        conn.execute("CREATE TABLE IF NOT EXISTS units (name TEXT UNIQUE)")
        conn.execute("INSERT OR IGNORE INTO units (name) VALUES (?)", (new_unit,))
        conn.commit()
        st.sidebar.success(f"✅ Added new unit: {new_unit}")
        st.rerun()

# === Section 选择 ===
section = st.sidebar.radio("📂 Sections", [
    "High Level report",
    "Sales report by category",
    "Inventory",
    "product mix",
    "Customers insights"
])

# === 主体展示 ===
if section == "High Level report":
    show_high_level(tx, mem, inv)
elif section == "Sales report by category":
    show_sales_report(tx, inv)
elif section == "Inventory":
    show_inventory(tx, inv)
elif section == "product mix":
    show_product_mix_only(tx)
elif section == "Customers insights":
    show_customer_segmentation(tx, mem)