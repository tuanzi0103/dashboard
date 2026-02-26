import re
import tempfile
from io import BytesIO
import datetime as _dt
import numpy as np

import pandas as pd
import streamlit as st
from services.db import get_db, init_database


# === Google Drive 相关 ===
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os
import time
from pathlib import Path
from contextlib import contextmanager


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

from services.db import get_db_path
from services.logger import init_logging, log_info, log_warning, log_error
init_logging()

import json

def sync_drive_with_database(drive_files):
    conn = get_db()

    # Drive 当前文件集合
    drive_set = set(drive_files)

    # 数据库记录的文件集合
    db_files = pd.read_sql("SELECT source_file FROM ingestion_log", conn)
    db_set = set(db_files["source_file"].tolist())

    # 被删除的文件
    deleted_files = db_set - drive_set

    for file in deleted_files:
        # 删除 transactions
        conn.execute("DELETE FROM transactions WHERE source_file = ?", (file,))

        # 删除 inventory
        conn.execute("DELETE FROM inventory WHERE source_file = ?", (file,))

        # 删除 log 记录
        conn.execute("DELETE FROM ingestion_log WHERE source_file = ?", (file,))

    conn.commit()
    conn.close()

def _expected_count_json_path(main_db: str) -> Path:
    """
    把 expected_drive_file_count 存在主库旁边的 json 文件里
    例如：/path/main.db  ->  /path/main.ingest_meta.json
    """
    p = Path(main_db)
    return p.with_suffix(".ingest_meta.json")


def load_expected_drive_file_count(main_db: str):
    """
    读取 json 里的 expected_drive_file_count
    返回 int 或 None
    """
    try:
        meta_path = _expected_count_json_path(main_db)
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        val = data.get("expected_drive_file_count")
        return int(val) if val is not None else None
    except Exception as e:
        log_warning(f"⚠️ Failed to read expected_drive_file_count json: {e}")
        return None


def save_expected_drive_file_count(main_db: str, count: int):
    """
    原子写入 json：先写 .tmp 再 replace，避免写到一半文件坏掉
    """
    try:
        meta_path = _expected_count_json_path(main_db)
        tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")

        payload = {
            "expected_drive_file_count": int(count),
            "updated_at": time.time(),
        }

        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp_path), str(meta_path))
        return True
    except Exception as e:
        log_warning(f"⚠️ Failed to save expected_drive_file_count json: {e}")
        try:
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False

def drive_get_content_file_with_retry(drive_file, local_path: str, retries: int = 3):
    # 退避：1s / 3s / 7s
    waits = [1, 3, 7]
    last_err = None
    for i in range(retries):
        try:
            drive_file.GetContentFile(local_path)
            return True
        except Exception as e:
            last_err = e
            wait = waits[i] if i < len(waits) else waits[-1]
            log_warning(f"⚠️ Drive download failed (attempt {i+1}/{retries}) for {local_path}: {e}")
            time.sleep(wait)
    raise RuntimeError(f"Drive download failed after {retries} retries: {local_path}") from last_err


@contextmanager
def ingest_file_lock(stale_seconds: int = 60 * 60):
    """
    跨进程文件锁（db/ingest.lock）
    - 同一时间只允许一个 ingest
    - stale_seconds 防止崩溃后死锁
    """
    db_path = Path(get_db_path()).resolve()
    db_dir = db_path.parent                 # ✅ 明确 db 目录
    lock_path = db_dir / "ingest.lock"      # ✅ 锁文件固定在 db/

    lock_fd = None

    def try_acquire():
        nonlocal lock_fd
        try:
            db_dir.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(
                str(lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
            os.write(
                lock_fd,
                f"pid={os.getpid()} time={time.time()}\n".encode()
            )
            return True
        except FileExistsError:
            return False

    acquired = try_acquire()

    # --- 处理陈旧锁 ---
    if not acquired and lock_path.exists():
        try:
            age = time.time() - lock_path.stat().st_mtime
            if age > stale_seconds:
                log_warning(f"⚠️ Stale ingest.lock detected ({int(age)}s), removing")
                lock_path.unlink(missing_ok=True)
                acquired = try_acquire()
        except Exception:
            pass

    if not acquired:
        yield False
        return

    try:
        log_info(f"🔒 Ingest lock acquired: {lock_path}")
        yield True
    finally:
        try:
            if lock_fd is not None:
                os.close(lock_fd)
        except Exception:
            pass
        try:
            lock_path.unlink(missing_ok=True)
            log_info("🔓 Ingest lock released")
        except Exception:
            pass


FOLDER_ID = "1lPmmJdB75yhDx2j4FxCjBW5iLZH3RQSp"

# ✅ 全局缓存 drive 实例
_drive_instance = None



def get_drive():
    """
    Fully robust Google Drive authentication:
    - Loads credentials.
    - If token expired → try refresh.
    - If refresh fails (invalid_grant) → delete token → force re-auth.
    - Always saves new token.
    """
    global _drive_instance
    if _drive_instance is not None:
        return _drive_instance

    gauth = GoogleAuth()

    # token 保存路径（pydrive2 默认推荐 token.json）
    TOKEN_PATH = "token.json"

    # 如果 token.json 不存在 → 强制首次登录
    if not os.path.exists(TOKEN_PATH):
        log_info("🔐 Please sign in to Google Drive.")
        gauth.LocalWebserverAuth()
        gauth.SaveCredentialsFile(TOKEN_PATH)
        _drive_instance = GoogleDrive(gauth)
        return _drive_instance

    # Step 1 — load existing credentials
    if os.path.exists(TOKEN_PATH):
        try:
            gauth.LoadCredentialsFile(TOKEN_PATH)
        except Exception:
            os.remove(TOKEN_PATH)
            gauth.credentials = None

    # Step 2 — If credentials exist, try refresh
    if gauth.credentials is not None:
        try:
            if gauth.access_token_expired:
                gauth.Refresh()
            else:
                gauth.Authorize()

        except Exception as e:
            # refresh failed → invalid_grant → must re-auth
            print("⚠️ Token refresh failed:", e)
            try:
                os.remove(TOKEN_PATH)
            except:
                pass
            gauth.LocalWebserverAuth()

    else:
        # No token at all → first-time login
        gauth.LocalWebserverAuth()

    # Step 3 — Save new token
    gauth.SaveCredentialsFile(TOKEN_PATH)

    _drive_instance = GoogleDrive(gauth)

    # --- log which account is authorized ---
    try:
        email = None

        # 1) preferred: id_token dict
        id_token = getattr(gauth.credentials, "id_token", None)
        if isinstance(id_token, dict):
            email = id_token.get("email")

        # 2) fallback: token_info (sometimes available)
        if not email:
            token_info = getattr(gauth.credentials, "token_info", None)
            if isinstance(token_info, dict):
                email = token_info.get("email")

        if email:
            log_info(f"🔐 Google Drive authorized as: {email}")
        else:
            log_info("🔐 Google Drive authorized (email not available in token).")

    except Exception as e:
        log_warning(f"⚠️ Could not read authorized email: {e}")

    return _drive_instance


def upload_file_to_drive(local_path: str, remote_name: str):
    """Upload file to Google Drive with success message."""
    try:
        drive = get_drive()  # now fully robust
        f = drive.CreateFile({'title': remote_name, 'parents': [{'id': FOLDER_ID}]})
        f.SetContentFile(local_path)
        f.Upload()
        log_info(f"☁️ Uploaded to Google Drive: {remote_name}")
        return True

    except Exception as e:
        log_warning(f"⚠️ Upload to Drive failed: {e}")
        return False


def download_file_from_drive(file_id, local_path):
    drive = get_drive()
    f = drive.CreateFile({'id': file_id})
    f.GetContentFile(local_path)


# --------------- 工具函数 ---------------

def _fix_header(df: pd.DataFrame) -> pd.DataFrame:
    """若第一行是 Unnamed，多数是多行表头；把第二行提为表头。"""
    if len(df.columns) and all(str(c).startswith("Unnamed") for c in df.columns):
        df.columns = df.iloc[0]
        df = df.drop(index=0).reset_index(drop=True)
    return df


def _to_float(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .replace("", pd.NA)
        .astype(float)
    )


def _extract_date_from_filename(name: str):
    """从文件名中提取 YYYY-MM-DD"""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if m:
        return m.group(1)
    return None

def _extract_date_range_from_filename(name: str):
    """
    支持 items-YYYY-MM-DD-YYYY-MM-DD.csv 这种，返回 (start_date, end_date) 字符串
    用于稳定排序 & 诊断“最早导入到哪一天”
    """
    m = re.findall(r"(\d{4}-\d{2}-\d{2})", name or "")
    if not m:
        return (None, None)
    if len(m) == 1:
        return (m[0], None)
    return (m[0], m[1])


def list_all_files_in_folder(drive, folder_id: str):
    """
    强制分页拉全量文件，避免只拿到“前 N 个”导致历史文件永远没导入。
    """
    q = f"'{folder_id}' in parents and trashed=false"
    params = {"q": q, "maxResults": 1000}
    all_files = []

    file_list = drive.ListFile(params)
    while True:
        batch = file_list.GetList()
        all_files.extend(batch)

        token = getattr(file_list, "metadata", {}).get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
        file_list = drive.ListFile(params)

    return all_files


# --------------- 预处理（不改列名） ---------------

def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = _fix_header(df)
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce"
        )
        drop_cols = [c for c in ["Date", "Time", "Time Zone"] if c in df.columns]
        df = df.drop(columns=drop_cols)
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    for col in ["Net Sales", "Gross Sales", "Qty", "Discounts"]:
        if col in df.columns:
            df[col] = _to_float(df[col])

    # === 新增：Card Brand 与 PAN Suffix 处理，保证写入数据库 ===
    if "Card Brand" in df.columns:
        df["Card Brand"] = (
            df["Card Brand"]
            .astype(str)
            .str.strip()
            .str.title()  # 标准化为首字母大写
        )

    if "PAN Suffix" in df.columns:
        df["PAN Suffix"] = (
            df["PAN Suffix"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)  # 去掉浮点形式的".0"
            .str.strip()
        )
    # === NEW: Clean item names / remove leading '*' ===
    for col in ["Item", "Item Name", "Price Point Name"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'^\*+', '', regex=True)  # remove one or more leading *
                .str.strip()
            )

    # === 自动分类：所有含“kombucha”关键字的项目归类为 Drinks ===

    if "Item" in df.columns and "Category" in df.columns:
        df["Item_lower"] = df["Item"].astype(str).str.lower()

        # 包含任何 "kombucha" 字样的 item → Drinks
        kombucha_mask = df["Item_lower"].str.contains("kombucha", na=False)

        df.loc[kombucha_mask, "Category"] = "Drinks"

        # 删除临时列
        df = df.drop(columns=["Item_lower"])

    return df


def preprocess_inventory(df: pd.DataFrame, filename: str = None) -> pd.DataFrame:
    df = _fix_header(df)

    # inventory表格从第二行开始是header
    if len(df) > 0 and all(str(col).startswith("Unnamed") for col in df.columns):
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        df = _fix_header(df)  # 再次处理可能的多行表头

    # === NEW: Clean leading '*' from Item/Variation columns ===
    for col in ["Item", "Item Name", "Variation Name", "SKU"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'^\*+', '', regex=True)  # 去掉开头的所有星号
                .str.strip()
            )

    required = [
        "Tax - GST (10%)", "Price", "Current Quantity Vie Market & Bar",
        "Default Unit Cost", "Categories"
    ]
    for col in required:
        if col not in df.columns:
            df[col] = None

    # 过滤掉Current Quantity Vie Market & Bar或者Default Unit Cost为空的行
    if "Current Quantity Vie Market & Bar" in df.columns and "Default Unit Cost" in df.columns:
        for col in ["Price", "Current Quantity Vie Market & Bar", "Default Unit Cost"]:
            if col not in df.columns:
                df[col] = None
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[^0-9\.\-]", "", regex=True)
                .replace("", pd.NA)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if filename:
        df["source_date"] = _extract_date_from_filename(filename)

    return df


def _is_transaction_file(filename: str, df: pd.DataFrame) -> bool:
    """双重判断：文件名 + 列名 才认为是交易文件"""
    fname = (filename or "").lower()

    # 文件名关键词
    name_ok = (
        "item" in fname or
        "transaction" in fname or
        "sales" in fname
    )

    # 列名关键词
    cols = {str(c).strip().lower() for c in df.columns}
    cols_ok = ("net sales" in cols and "gross sales" in cols)

    return name_ok and cols_ok


def _is_inventory_file(filename: str, df: pd.DataFrame) -> bool:
    """双重判断：文件名 + 列名 才认为是库存文件"""
    fname = (filename or "").lower()

    # 文件名关键词
    name_ok = (
        "catalogue" in fname or
        "inventory" in fname or
        "stock" in fname
    )

    # 列名关键词：至少出现任意一个
    cols = {str(c).strip().lower() for c in df.columns}
    cols_ok = (
        "sku" in cols or
        "categories" in cols or
        "stock on hand" in cols
    )

    return name_ok and cols_ok


def _is_member_file(filename: str, df) -> bool:
    """
    判断一个文件是否为 member 文件：
    - 文件名包含 'member'（不分大小写），或者
    - 列名中同时包含 First Name / Surname / Birthday （不分大小写）
    """
    fname = (filename or "").lower()
    has_member_in_name = "member" in fname

    # df 不是 DataFrame 的时候，不要直接访问 .columns，以免报错
    if not hasattr(df, "columns"):
        # 退一步：如果文件名里写了 member，就当成 member 文件，否则直接 False
        return has_member_in_name

    cols_lower = {str(c).strip().lower() for c in df.columns}
    has_core_cols = {"first name", "surname", "birthday"}.issubset(cols_lower)

    return has_member_in_name or has_core_cols



def preprocess_members(df: pd.DataFrame) -> pd.DataFrame:
    df = _fix_header(df)

    # 1) 标准化列名：全部转小写后做映射
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()

        if cl == "surname":
            rename_map[c] = "Last Name"
        elif cl == "last name":
            rename_map[c] = "Last Name"
        elif cl == "first name":
            rename_map[c] = "First Name"
        elif cl == "square customer id":
            rename_map[c] = "Square Customer ID"
        elif cl == "email address":
            rename_map[c] = "Email Address"
        elif cl == "phone number":
            rename_map[c] = "Phone Number"
        elif cl == "creation date":
            rename_map[c] = "Creation Date"
        elif cl == "customer note":
            rename_map[c] = "Customer Note"
        elif cl == "reference id":
            rename_map[c] = "Reference ID"

    df = df.rename(columns=rename_map)

    # 2) 清理 Phone Number 字段 - 移除注释文本
    if "Phone Number" in df.columns:
        def clean_phone(phone):
            if pd.isna(phone):
                return phone
            phone_str = str(phone)
            # 查找手机号模式：以+61或61开头
            import re
            match = re.search(r'(\+?61\d{8,9})', phone_str)
            if match:
                return match.group(1)
            # 如果没有找到，返回原始值
            return phone_str

        df["Phone Number"] = df["Phone Number"].apply(clean_phone)

    # 3) 清理 Square Customer ID 和 Reference ID
    for col in ["Square Customer ID", "Reference ID"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 4) 只保留和 DB 对齐的列
    allowed_cols = [
        "Square Customer ID",
        "First Name",
        "Last Name",
        "Email Address",
        "Phone Number",
        "Creation Date",
        "Customer Note",
        "Reference ID",
    ]
    existing = [c for c in df.columns if c in allowed_cols]
    df = df[existing]

    # 5) 简单清洗：去空格
    for col in ["Square Customer ID", "First Name", "Last Name",
                "Email Address", "Phone Number", "Reference ID"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


# --------------- 表结构对齐 & 去重 & 写入 ---------------

def _table_exists(conn, table: str) -> bool:
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return cur.fetchone() is not None
    except Exception:
        return False


def _existing_columns(conn, table: str) -> list:
    try:
        cur = conn.execute(f"PRAGMA table_info('{table}')")
        return [row[1] for row in cur.fetchall()]
    except Exception:
        return []


def _add_missing_columns(conn, table: str, missing_cols: list, prefer_real: set):
    cur = conn.cursor()
    for col in missing_cols:
        coltype = "REAL" if col in prefer_real else "TEXT"
        cur.execute(f'''ALTER TABLE "{table}" ADD COLUMN "{col}" {coltype}''')
    conn.commit()


def _ensure_table_schema(conn, table: str, df: pd.DataFrame, prefer_real: set):
    if not _table_exists(conn, table):
        # 如果表不存在，创建表
        df.head(0).to_sql(table, conn, if_exists="replace", index=False)
        return
    cols_now = set(_existing_columns(conn, table))
    incoming = list(df.columns)
    missing = [c for c in incoming if c not in cols_now]
    if missing:
        _add_missing_columns(conn, table, missing, prefer_real)


def _deduplicate(df: pd.DataFrame, key_col: str, conn, table: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    if table == "inventory" and "source_date" in df.columns and "SKU" in df.columns:
        try:
            # 只与同一天的数据去重
            exist = pd.read_sql('SELECT source_date, SKU FROM "inventory"', conn)
            exist["source_date"] = pd.to_datetime(exist["source_date"], errors="coerce").dt.date.astype(str)
            exist["SKU"] = exist["SKU"].astype(str)

            df_local = df.copy()
            df_local["source_date"] = pd.to_datetime(df_local["source_date"], errors="coerce").dt.date.astype(str)
            df_local["SKU"] = df_local["SKU"].astype(str)

            # ✅ 只与相同日期比对，而非所有日期
            existed_keys = set((exist["source_date"] + "||" + exist["SKU"]).unique())
            keys = df_local["source_date"] + "||" + df_local["SKU"]

            mask = ~keys.isin(existed_keys)
            return df_local[mask]
        except Exception:
            return df

    # 其它表/场景：保持原单键去重逻辑
    if key_col not in df.columns:
        return df
    try:
        exist = pd.read_sql(f'''SELECT "{key_col}" FROM "{table}"''', conn)
        existed_set = set(exist[key_col].dropna().astype(str).unique())
        mask = ~df[key_col].astype(str).isin(existed_set)
        return df[mask]
    except Exception:
        return df


def _sqlite_sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """把 pandas / numpy 类型转换成 sqlite3 支持的参数类型。"""
    out = df.copy()

    # 1) datetime64 列 -> 字符串
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    # 2) object 列里可能混进 pandas.Timestamp（尤其 Datetime 被预处理成 object 时）
    def fix_cell(x):
        if x is None:
            return None

        # NaN / NaT
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass

        # pandas.Timestamp
        if isinstance(x, pd.Timestamp):
            if pd.isna(x):
                return None
            return x.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")

        # numpy.datetime64
        if isinstance(x, np.datetime64):
            try:
                ts = pd.to_datetime(x, errors="coerce")
                if pd.isna(ts):
                    return None
                return ts.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return None

        # datetime.datetime
        if isinstance(x, _dt.datetime):
            return x.strftime("%Y-%m-%d %H:%M:%S")

        # datetime.date（注意：date 没有时间）
        if isinstance(x, _dt.date):
            return _dt.datetime(x.year, x.month, x.day).strftime("%Y-%m-%d %H:%M:%S")

        return x

    out = out.applymap(fix_cell)

    return out

def _write_df(conn, df: pd.DataFrame, table: str, key_candidates: list, prefer_real: set, is_initial_build: bool = False):
    """
    统一写入数据库：
    - inventory：同一天先删再写（保留你原逻辑）
    - members：增删改（保留你原逻辑）
    - transactions：✅ 用行级复合 key 去重（根治重叠文件吞历史）
    返回：实际写入（去重后）行数 inserted
    """
    if df is None or df.empty:
        return 0

    # --- 通用：确保表结构一致 ---
    _ensure_table_schema(conn, table, df, prefer_real)

    # ----------------------------------------------------------------------
    # inventory：同一天先删除再写入
    # ----------------------------------------------------------------------
    if table == "inventory" and "source_date" in df.columns:
        dates = df["source_date"].dropna().unique().tolist()
        if dates:
            for d in dates:
                conn.execute(f'DELETE FROM "{table}" WHERE source_date=?', (d,))
            conn.commit()

    # ----------------------------------------------------------------------
    # members：增删改（保留你原逻辑，最后 return inserted）
    # ----------------------------------------------------------------------
    if table == "members":
        df = _sqlite_sanitize_df(df)
        try:
            df_old = pd.read_sql('SELECT * FROM "members"', conn)
        except Exception:
            df_old = pd.DataFrame()

        key = None
        for k in ["Square Customer ID", "Reference ID"]:
            if k in df.columns:
                key = k
                break
        if key is None:
            return 0

        df[key] = df[key].astype(str)
        if not df_old.empty:
            df_old[key] = df_old[key].astype(str)

        old_keys = set(df_old[key]) if not df_old.empty else set()
        new_keys = set(df[key])

        keys_to_delete = old_keys - new_keys
        if keys_to_delete:
            placeholders = ",".join(["?"] * len(keys_to_delete))
            conn.execute(
                f'DELETE FROM "members" WHERE "{key}" IN ({placeholders})',
                tuple(keys_to_delete)
            )

        keys_to_insert = new_keys - old_keys
        df_insert = df[df[key].isin(keys_to_insert)]
        inserted = int(len(df_insert))
        if not df_insert.empty:
            # ✅ 1) 写入前再 sanitize 一次（双保险）
            df_insert = _sqlite_sanitize_df(df_insert)
            df_insert = df_insert.where(pd.notnull(df_insert), None)
            df_insert.to_sql("members", conn, if_exists="append", index=False)

        # 🚀 第一次建库：不要逐行 UPDATE（极慢），只插入/删除即可
        if (not is_initial_build) and (not df_old.empty):
            df_merge = df.merge(df_old, on=key, how="inner", suffixes=("_new", "_old"))
            update_cols = [c for c in df.columns if c != key]
            for _, row in df_merge.iterrows():
                changed = any(str(row[f"{c}_new"]) != str(row[f"{c}_old"]) for c in update_cols)
                if changed:
                    set_clause = ", ".join([f'"{c}"=?' for c in update_cols])
                    params = [row[f"{c}_new"] for c in update_cols] + [row[key]]
                    params = _sqlite_sanitize_df(pd.DataFrame([params])).iloc[0].tolist()
                    params = [None if (isinstance(x, float) and pd.isna(x)) else x for x in params]

                    conn.execute(
                        f'UPDATE members SET {set_clause} WHERE "{key}"=?',
                        params
                    )

        conn.commit()
        return inserted

    if table == "transactions":
        # 1) 保证这些列都存在
        needed = [
            "Transaction ID", "Datetime", "Item", "Net Sales", "Gross Sales",
            "Discounts", "Qty", "Customer ID", "Modifiers Applied",
            "Tax", "Card Brand", "PAN Suffix"
        ]
        for c in needed:
            if c not in df.columns:
                df[c] = ""

        df_local = df.copy()

        # 2) 先做一个“行内容 base”（尽量包含完整信息）
        base_cols = [
            "Transaction ID", "Datetime", "Item", "Net Sales", "Gross Sales",
            "Discounts", "Qty", "Customer ID", "Modifiers Applied",
            "Tax", "Card Brand", "PAN Suffix"
        ]
        for c in ["Net Sales", "Gross Sales", "Discounts", "Qty"]:
            if c in df_local.columns:
                df_local[c] = pd.to_numeric(df_local[c], errors="coerce").fillna(0)
                # 金额统一保留2位，Qty 可以保留3位或不 round
                if c != "Qty":
                    df_local[c] = df_local[c].round(2)

        df_local["__base"] = df_local[base_cols].astype(str).agg("||".join, axis=1)

        # 3) 对于完全相同的行内容，在同一 Transaction 内给一个稳定的序号
        #    这样同一小票里重复的两行不会互相吞
        df_local = df_local.sort_values(
            ["Transaction ID", "Datetime", "Item", "Net Sales", "Qty", "Customer ID", "__base"],
            kind="mergesort"
        )
        df_local["__dup_idx"] = df_local.groupby(["Transaction ID", "__base"]).cumcount()

        # 4) 最终 row_key = base + dup_idx
        df_local["__row_key"] = df_local["__base"] + "||" + df_local["__dup_idx"].astype(str)

        # 只保留 row_key，删掉临时列
        df_local = df_local.drop(columns=["__base", "__dup_idx"])

        # ✅确保列存在（正式库 / tmp库都要有这列）
        try:
            conn.execute('ALTER TABLE "transactions" ADD COLUMN "__row_key" TEXT')
        except Exception:
            pass

        # ✅唯一索引：防止重复写入
        try:
            conn.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_tx_row_key ON transactions("__row_key")')
        except Exception:
            pass

        # 用 DB 唯一索引兜底，不依赖 _deduplicate 成功与否
        cols = list(df_local.columns)
        placeholders = ",".join(["?"] * len(cols))
        col_sql = ",".join([f'"{c}"' for c in cols])
        sql = f'INSERT OR IGNORE INTO "transactions" ({col_sql}) VALUES ({placeholders})'

        # ✅ 关键：写入前把 Timestamp / NaT / NaN 清掉
        df_local = _sqlite_sanitize_df(df_local)
        df_local = df_local.where(pd.notnull(df_local), None)

        before = conn.total_changes
        conn.executemany(sql, df_local.itertuples(index=False, name=None))
        conn.commit()
        inserted = conn.total_changes - before
        return int(inserted)

    # ----------------------------------------------------------------------
    # 其它表：保留原单键去重
    # ----------------------------------------------------------------------
    key_col = next((k for k in key_candidates if k in df.columns), None)
    if key_col:
        df = _deduplicate(df, key_col, conn, table)

    inserted = int(len(df))
    if inserted > 0:
        df.to_sql(table, conn, if_exists="append", index=False)

    return inserted



# --------------- 索引 ---------------
def ensure_indexes():
    """Ensure DB schema + indexes exist.

    IMPORTANT:
    - Do NOT create empty placeholder tables with pandas (it may create tables with 0 columns).
    - Always call init_database() which creates the correct schema and indexes.
    """
    try:
        init_database()
    except Exception as e:
        # Do not crash ingestion for index creation issues; just log.
        try:
            log_error(f"❌ init_database() failed in ensure_indexes(): {e}")
        except Exception:
            pass


def ingest_from_drive_all():
    is_initial_build = not Path(get_db_path()).exists()
    with ingest_file_lock() as locked:
        if not locked:
            log_warning("⏳ ingest already running (file lock), skip ingest_from_drive_all()")
            return False
        _ingest_from_drive_all_impl(is_initial_build=is_initial_build)
        return True


# --------------- 从 Google Drive 导入 ---------------
def _ingest_from_drive_all_impl(is_initial_build: bool = False):
    """
    ✅ 最终稳态：临时库 + 成功后替换正式库
    - 先把所有数据导入到 main_db.tmp
    - 导入“足够完整”才用 tmp 原子替换 main_db
    - 任意中途失败/半残导入：丢弃 tmp，不破坏现有正式库
    """
    import shutil
    import sqlite3

    # --- 0) 找到正式库路径（不依赖 services/db.py 增加新函数）---
    main_conn = get_db()
    try:
        row = main_conn.execute("PRAGMA database_list").fetchone()
        # row 通常是 (seq, name, file)
        main_db = row[2] if row and len(row) >= 3 else None
    finally:
        try:
            main_conn.close()
        except Exception:
            pass

    if not main_db:
        log_error("❌ Cannot resolve main DB path (PRAGMA database_list empty). Abort ingest.")
        return

    tmp_db = main_db + ".tmp"
    bak_db = main_db + ".bak"

    # --- 1) 清理旧 tmp ---
    try:
        if os.path.exists(tmp_db):
            os.remove(tmp_db)
    except Exception:
        pass

    tmp_conn = sqlite3.connect(tmp_db)
    tmp_conn.row_factory = sqlite3.Row

    # 🚀 大幅提速：建库阶段用更快的写入参数（tmp库安全）
    tmp_conn.execute("PRAGMA synchronous = OFF;")
    tmp_conn.execute("PRAGMA journal_mode = MEMORY;")
    tmp_conn.execute("PRAGMA temp_store = MEMORY;")
    tmp_conn.execute("PRAGMA cache_size = -200000;")  # ~200MB cache（可按需调小）
    tmp_conn.execute("PRAGMA locking_mode = EXCLUSIVE;")

    cur = tmp_conn.cursor()

    # 用你 services/db.py 里相同结构建表（最小复制，保证一致）
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            Datetime TEXT,
            Category TEXT,
            Item TEXT,
            Qty REAL,
            [Net Sales] REAL,
            [Gross Sales] REAL,
            Discounts REAL,
            [Customer ID] TEXT,
            [Transaction ID] TEXT,
            Tax TEXT,
            [Card Brand] TEXT,
            [PAN Suffix] TEXT,
            [Date] TEXT,
            [Time] TEXT,
            [Time Zone] TEXT,
            [Modifiers Applied] TEXT,
            __row_key TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_tx_row_key ON transactions(__row_key);
        CREATE TABLE IF NOT EXISTS inventory (
            [Product ID] TEXT,
            [Product Name] TEXT,
            SKU TEXT,
            Categories TEXT,
            Price REAL,
            [Tax - GST (10%)] TEXT,
            [Current Quantity Vie Market & Bar] REAL,
            [Default Unit Cost] REAL,
            Unit TEXT,
            source_date TEXT,
            [Stock on Hand] REAL
        );

        CREATE TABLE IF NOT EXISTS members (
            [Square Customer ID] TEXT,
            [First Name] TEXT,
            [Last Name] TEXT,
            [Email Address] TEXT,
            [Phone Number] TEXT,
            [Creation Date] TEXT,
            [Customer Note] TEXT,
            [Reference ID] TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_txn_datetime ON transactions(Datetime);
        CREATE INDEX IF NOT EXISTS idx_txn_id ON transactions([Transaction ID]);
        CREATE INDEX IF NOT EXISTS idx_inv_sku ON inventory(SKU);
        CREATE INDEX IF NOT EXISTS idx_inv_categories ON inventory(Categories);
        CREATE INDEX IF NOT EXISTS idx_member_square ON members([Square Customer ID]);
        CREATE INDEX IF NOT EXISTS idx_member_ref ON members([Reference ID]);
    """)
    tmp_conn.commit()

    # --- 3) 拉 Drive 文件列表（全量分页），并做“列表完整性”保护 ---
    drive = get_drive()
    files = list_all_files_in_folder(drive, FOLDER_ID)
    log_info(f"📦 Drive files fetched: {len(files)}")
    # --- 历史文件数完整性校验 ---
    expected_count = load_expected_drive_file_count(main_db)
    if (not is_initial_build) and expected_count:
        current_count = len(files)
        # 容忍 10% 波动
        if current_count < int(expected_count * 0.9):
            log_warning(
                f"🛑 Drive file count too small: {current_count} < 90% of expected {expected_count}. Abort ingest."
            )
            try:
                tmp_conn.close()
            except Exception:
                pass
            try:
                if os.path.exists(tmp_db):
                    os.remove(tmp_db)
            except Exception:
                pass
            return

    if not files:
        log_warning("⚠️ No files found in Drive folder.")
        try:
            tmp_conn.close()
        except Exception:
            pass
        try:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)
        except Exception:
            pass
        return

    # --- 4) 稳定排序（确定性，跨机器一致） ---
    def sort_key(f):
        name = f.get("title") or ""
        file_id = f.get("id") or ""
        start_date, _ = _extract_date_range_from_filename(name)

        sd_ts = pd.to_datetime(start_date, errors="coerce") if start_date else pd.NaT
        sd_val = int(sd_ts.value) if not pd.isna(sd_ts) else -1
        missing = 1 if sd_val == -1 else 0
        return (missing, -sd_val, name.strip().lower(), file_id)

    files = sorted(files, key=sort_key)

    # 诊断：打印日期跨度
    dates = []
    for f in files:
        name = f.get("title") or ""
        sd, _ = _extract_date_range_from_filename(name)
        if sd:
            dates.append(sd)
    if dates:
        log_info(f"🧭 File date span (by filename): {min(dates)}  →  {max(dates)}")

    seen = set()
    error_files = []
    attempted_supported = 0
    succeeded_supported = 0

    # --- 5) 开始导入到临时库 ---
    for f in files:
        name = f.get("title") or ""
        if not name:
            continue

        if name in seen:
            continue
        seen.add(name)

        local = os.path.join(tempfile.gettempdir(), name)

        try:
            is_csv = name.lower().endswith(".csv")
            is_xlsx = name.lower().endswith(".xlsx")

            # 先统计“支持类型文件总数”
            if is_csv or is_xlsx:
                attempted_supported += 1
            else:
                log_warning(f"⚠️ Skip unsupported file: {name}")
                continue

            # 下载（带重试）
            drive_get_content_file_with_retry(f, local, retries=3)

            # 读取
            if is_csv:
                df = pd.read_csv(local)
            else:
                header_row = 1 if "catalogue" in name.lower() else 0
                df = pd.read_excel(local, header=header_row)

            df = _fix_header(df)

            # 判断类型 & 写入临时库
            if _is_transaction_file(name, df):
                df = preprocess_transactions(df)

                if "Datetime" in df.columns:
                    dt = pd.to_datetime(df["Datetime"], errors="coerce")
                    nat_rate = float(dt.isna().mean())
                    if nat_rate >= 0.8:
                        log_error(f"❌ TX {name}: Datetime parse failed (NaT={nat_rate:.1%}), skipped")
                        error_files.append((name, f"Datetime NaT {nat_rate:.1%}"))
                        continue
                else:
                    log_error(f"❌ TX {name}: missing Datetime column, skipped")
                    error_files.append((name, "Missing Datetime column"))
                    continue

                inserted = _write_df(
                    tmp_conn, df, "transactions",
                    key_candidates=["Transaction ID", "Item", "Price", "Modifiers Applied"],
                    prefer_real={"Net Sales", "Gross Sales", "Qty", "Discounts"},
                    is_initial_build=is_initial_build
                )

                succeeded_supported += 1
                if inserted == 0:
                    log_warning(f"⚠️ TX {name}: inserted=0 (likely deduped)")
                log_info(f"✅ TX {name}: read={len(df)} inserted={inserted} NaT={nat_rate:.1%}")

            elif _is_inventory_file(name, df):
                df = preprocess_inventory(df, filename=name)
                inserted = _write_df(
                    tmp_conn, df, "inventory",
                    key_candidates=["SKU"], prefer_real=set(),
                    is_initial_build=is_initial_build
                )

                succeeded_supported += 1
                if inserted == 0:
                    log_warning(f"⚠️ INV {name}: inserted=0 (likely deduped)")
                log_info(f"✅ INV {name}: read={len(df)} inserted={inserted}")

            elif _is_member_file(name, df):
                try:
                    df_members = preprocess_members(df)
                    inserted = _write_df(
                        tmp_conn, df_members, "members",
                        key_candidates=["Square Customer ID", "Reference ID"],
                        prefer_real=set(),
                        is_initial_build=is_initial_build
                    )

                    succeeded_supported += 1
                    log_info(f"📥 Members {name}: read={len(df_members)} inserted={inserted}")
                except Exception as e:
                    error_files.append((name, str(e)))
                    log_error(f"❌ Failed to import members from {name}: {e}")

            else:
                log_warning(f"⚠️ Schema not recognized, skipped: {name}")
                # 识别不了不算“成功导入”，但也不算错误
                continue

        except Exception as e:
            msg = str(e)
            error_files.append((name, msg))
            log_error(f"❌ Failed to import {name}: {msg}")

        finally:
            try:
                if os.path.exists(local):
                    os.remove(local)
            except Exception:
                pass

    # --- 6) 完整性检查：避免半残 tmp 覆盖正式库 ---
    try:
        tx_rows = tmp_conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    except Exception:
        tx_rows = 0

    # 规则 A：成功率阈值（避免只导入少量 supported 文件）
    # attempted_supported 是 .csv/.xlsx 的数量；成功率过低说明中途失败或列表不完整
    success_rate = (succeeded_supported / attempted_supported) if attempted_supported else 0.0
    MIN_SUCCESS_RATE = 0.90  # 建议会前先 0.90，稳定后可以升到 0.95

    # 规则 B：最早日期对齐校验（关键）
    # - min_file_date：从文件名解析到的最早日期（只看能解析出日期的文件）
    # - min_tx_date：tmp 库 transactions 的最早 Datetime 日期
    min_file_date = None
    try:
        parsed_dates = []
        for f in files:
            name = f.get("title") or ""
            sd, _ = _extract_date_range_from_filename(name)
            if sd:
                d = pd.to_datetime(sd, errors="coerce")
                if not pd.isna(d):
                    parsed_dates.append(d.normalize())
        if parsed_dates:
            min_file_date = min(parsed_dates)
    except Exception:
        min_file_date = None

    min_tx_date = None
    try:
        # Datetime 存的是字符串，这里用 sqlite 的 date() 抽取日期
        row = tmp_conn.execute("SELECT MIN(date(Datetime)) FROM transactions").fetchone()
        if row and row[0]:
            min_tx_date = pd.to_datetime(row[0], errors="coerce").normalize()
    except Exception:
        min_tx_date = None

    log_info(
        f"📊 TMP DB check: tx_rows={tx_rows}, attempted_files={attempted_supported}, "
        f"succeeded_files={succeeded_supported}, success_rate={success_rate:.1%}, "
        f"min_file_date={min_file_date.date() if min_file_date is not None else None}, "
        f"min_tx_date={min_tx_date.date() if min_tx_date is not None else None}"
    )

    # ✅ 判定：如果能解析出 min_file_date，但 tmp 的最早交易日期明显晚于它（容差 1 天），说明漏了早期大段文件
    DATE_TOLERANCE_DAYS = 1
    date_ok = True
    if min_file_date is not None and min_tx_date is not None:
        date_ok = (min_tx_date <= (min_file_date + pd.Timedelta(days=DATE_TOLERANCE_DAYS)))

    # ✅ 最终 gate：成功率 + 日期对齐
    if (success_rate < MIN_SUCCESS_RATE) or (not date_ok):
        reason = []
        if success_rate < MIN_SUCCESS_RATE:
            reason.append(f"success_rate<{MIN_SUCCESS_RATE:.0%}")
        if not date_ok:
            reason.append(f"min_tx_date>{DATE_TOLERANCE_DAYS}d after min_file_date")
        log_warning(
            f"🛑 TMP DB aborted summary: "
            f"attempted_supported={attempted_supported}, "
            f"succeeded_supported={succeeded_supported}, "
            f"success_rate={success_rate:.1%}, "
            f"min_file_date={min_file_date.date() if min_file_date is not None else None}, "
            f"min_tx_date={min_tx_date.date() if min_tx_date is not None else None}"
        )

        try:
            tmp_conn.close()
        except Exception:
            pass
        try:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)
        except Exception:
            pass
        return

    # --- 7) 原子替换：tmp -> main（并备份） ---
    try:
        tmp_conn.close()
    except Exception:
        pass

    try:
        if os.path.exists(bak_db):
            os.remove(bak_db)
    except Exception:
        pass

    try:
        if os.path.exists(main_db):
            shutil.copy2(main_db, bak_db)  # 备份旧库
        os.replace(tmp_db, main_db)  # 原子替换（同盘）
        log_info("✅ TMP DB committed: replaced main DB successfully.")

        # --- 记录“历史正常文件数”作为基准（写 db 旁边 json，不打开主库） ---
        ok = save_expected_drive_file_count(main_db, len(files))
        if ok:
            log_info(f"📌 Saved expected_drive_file_count={len(files)} (json next to db)")


    except Exception as e:
        log_error(f"❌ Failed to replace main DB: {e}")
        # 替换失败：保留旧库，清理 tmp
        try:
            if os.path.exists(tmp_db):
                os.remove(tmp_db)
        except Exception:
            pass

    # --- 8) 打印失败文件汇总（不影响成功提交） ---
    if error_files:
        log_warning("⚠️ Some Drive files were skipped when building database:")
        for fname, _ in error_files:
            log_info(f"• {fname}")


def init_db_from_drive_once():
    """
    自动初始化数据库（仅在库为空时）
    - 使用文件锁，避免并发
    - 不再使用 ingest_meta 表
    """
    with ingest_file_lock() as locked:
        if not locked:
            log_warning("⏳ ingest already running (file lock), skip init_db_from_drive_once()")
            try:
                st.info("Database is initializing in another process. Please wait a moment.")
            except Exception:
                pass
            return False

        try:
            conn = get_db()
            cur = conn.cursor()

            try:
                tx_count = cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
                inv_count = cur.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]
            except Exception:
                tx_count, inv_count = 0, 0

            conn.close()

            # 只有在“真正空库”时才 ingest
            if tx_count == 0 and inv_count == 0:
                log_info("🚀 Empty DB detected, ingesting from Drive...")
                _ingest_from_drive_all_impl()

            return True

        except Exception as e:
            log_warning(f"⚠️ Auto-ingest from Drive failed: {e}")
            return False




# --------------- 手动导入（Sidebar 上传） ---------------
def ingest_csv(uploaded_file, source_file=None):
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    filename = uploaded_file.name if hasattr(uploaded_file, "name") else "uploaded.csv"
    conn = get_db()
    ensure_indexes()
    log_info(f"📂 Importing {filename}")

    try:
        df = pd.read_csv(BytesIO(data))
        df = _fix_header(df)

        if _is_transaction_file(filename, df):
            df = preprocess_transactions(df)
            inserted = _write_df(conn, df, "transactions",
                                 key_candidates=["Transaction ID"],
                                 prefer_real={"Net Sales", "Gross Sales", "Qty", "Discounts"})

        elif _is_inventory_file(filename, df):
            df = preprocess_inventory(df, filename=filename)
            inserted = _write_df(conn, df, "inventory",
                                 key_candidates=["SKU"], prefer_real=set())

        elif _is_member_file(filename, df):
            df = preprocess_members(df)
            inserted = _write_df(conn, df, "members",
                                 key_candidates=["Square Customer ID", "Reference ID"], prefer_real=set())
        else:
            log_warning(f"⚠️ Skipped {filename}, schema not recognized")
            return False
        # === NEW: record ingested file ===
        conn.execute(
            """
            INSERT OR IGNORE INTO ingestion_log (source_file)
            VALUES (?)
            """,
            (filename,)
        )

        # 上传到 Google Drive（用同一份 data，永不为空）
        tmp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(tmp_path, "wb") as f_local:
            f_local.write(data)

        uploaded_drive_files = st.session_state.get("uploaded_drive_files", set())
        if filename not in uploaded_drive_files:
            upload_file_to_drive(tmp_path, filename)
            uploaded_drive_files.add(filename)
        st.session_state["uploaded_drive_files"] = uploaded_drive_files

        ensure_indexes()
        return True

    except Exception as e:
        log_error(f"❌ Error importing {filename}: {str(e)}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            if 'tmp_path' in locals():
                os.remove(tmp_path)
        except Exception:
            pass


def ingest_excel(uploaded_file):
    conn = get_db()
    ensure_indexes()

    filename = uploaded_file.name if hasattr(uploaded_file, "name") else "uploaded.xlsx"
    log_info(f"📂 Importing {filename}")

    try:
        data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        xls = pd.ExcelFile(BytesIO(data))

        total_rows_imported = 0

        is_catalogue = ("catalogue" in filename.lower())
        # 只处理 Items/首个 sheet（库存）
        if is_catalogue:
            target_sheets = []
            if "Items" in xls.sheet_names:
                target_sheets = ["Items"]
            else:
                # 回退：找第一个 sheet
                target_sheets = [xls.sheet_names[0]]

            inv_frames = []
            for sheet in target_sheets:
                df = pd.read_excel(xls, sheet_name=sheet, header=1)
                df = _fix_header(df)
                if ("SKU" in df.columns) or ("Stock on Hand" in df.columns) or ("Categories" in df.columns):
                    df = preprocess_inventory(df, filename=filename)
                    inv_frames.append(df)

            if inv_frames:
                inv_all = pd.concat(inv_frames, ignore_index=True)
                inserted = _write_df(conn, inv_all, "inventory",
                                     key_candidates=["SKU"], prefer_real=set())
                total_rows_imported += inserted

        else:
            # 非 catalogue 的 Excel：保留原逻辑（逐 sheet 导入）
            for sheet in xls.sheet_names:
                header_row = 0
                df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
                df = _fix_header(df)

                if _is_transaction_file(filename, df):
                    df = preprocess_transactions(df)
                    inserted = _write_df(conn, df, "transactions",
                                         key_candidates=["Transaction ID"],
                                         prefer_real={"Net Sales", "Gross Sales", "Qty", "Discounts"})
                    total_rows_imported += inserted


                elif _is_inventory_file(filename, df):
                    df = preprocess_inventory(df, filename=filename)
                    inserted = _write_df(conn, df, "inventory",
                                         key_candidates=["SKU"], prefer_real=set())
                    total_rows_imported += inserted

                elif _is_member_file(filename, df):
                    df = preprocess_members(df)
                    inserted = _write_df(conn, df, "members",
                                         key_candidates=["Square Customer ID", "Reference ID"], prefer_real=set())
                    total_rows_imported += inserted

        log_info(f"✅ {filename} imported - {total_rows_imported} total rows")

        # === NEW: record ingested file ===
        conn.execute(
            """
            INSERT OR IGNORE INTO ingestion_log (source_file)
            VALUES (?)
            """,
            (filename,)
        )


        # 上传到 Drive（保持原逻辑）
        tmp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(tmp_path, "wb") as f_local:
            f_local.write(data)
        # === 防止重复上传到 Google Drive ===
        uploaded_drive_files = st.session_state.get("uploaded_drive_files", set())
        if filename not in uploaded_drive_files:
            upload_file_to_drive(tmp_path, filename)
            uploaded_drive_files.add(filename)
        st.session_state["uploaded_drive_files"] = uploaded_drive_files

        ensure_indexes()
        return True

    except Exception as e:
        log_error(f"❌ Error importing {filename}: {str(e)}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            if 'tmp_path' in locals():
                os.remove(tmp_path)
        except Exception:
            pass

def ingest_new_files_from_drive_only():
    """
    只 ingest Google Drive 中「还没进过数据库」的新文件
    不 rebuild，不清库
    """
    from services.db import get_db

    conn = get_db()
    init_database()

    # 1. 取数据库里已有的文件名（用于去重）
    existing_files = set()

    try:
        rows = conn.execute("""
            SELECT DISTINCT source_file
            FROM ingestion_log
        """).fetchall()
        existing_files = {r[0] for r in rows if r[0]}
    except Exception:
        # 第一次跑可能还没 ingestion_log 表
        pass

    drive = get_drive()
    files = list_all_files_in_folder(drive, FOLDER_ID)

    new_files = []

    for f in files:
        name = f.get("title", "").lower()
        if not (name.endswith(".csv") or name.endswith(".xlsx")):
            continue

        if f["title"] in existing_files:
            continue  # 已 ingest，跳过

        new_files.append(f)

    if not new_files:
        return 0  # 没有新文件

    # 2. ingest 新文件
    import tempfile, os, pandas as pd

    for f in new_files:
        file_id = f["id"]
        filename = f["title"]
        local = os.path.join(tempfile.gettempdir(), filename)

        drive_file = drive.CreateFile({'id': file_id})
        drive_get_content_file_with_retry(drive_file, local)

        # === 读取文件 ===
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(local)
        else:
            header_row = 1 if "catalogue" in filename.lower() else 0
            df = pd.read_excel(local, header=header_row)

        df = _fix_header(df)

        # === 判断类型并写入主库（和全量 ingest 同一逻辑） ===
        if _is_transaction_file(filename, df):
            df = preprocess_transactions(df)
            _write_df(
                conn, df, "transactions",
                key_candidates=["Transaction ID"],
                prefer_real={"Net Sales", "Gross Sales", "Qty", "Discounts"},
                is_initial_build=False
            )

        elif _is_inventory_file(filename, df):
            df = preprocess_inventory(df, filename=filename)
            _write_df(
                conn, df, "inventory",
                key_candidates=["SKU"],
                prefer_real=set(),
                is_initial_build=False
            )

        elif _is_member_file(filename, df):
            df = preprocess_members(df)
            _write_df(
                conn, df, "members",
                key_candidates=["Square Customer ID", "Reference ID"],
                prefer_real=set(),
                is_initial_build=False
            )
        else:
            log_warning(f"⚠️ Drive file skipped (schema not recognized): {filename}")
            continue

        # === 记录 ingestion_log（关键） ===
        conn.execute(
            """
            INSERT OR IGNORE INTO ingestion_log (source_file)
            VALUES (?)
            """,
            (filename,)
        )

    try:
        conn.commit()
        return len(new_files)
    finally:
        try:
            conn.close()
        except Exception:
            pass


__all__ = [
    "ingest_csv",
    "ingest_excel",
    "ingest_from_drive_all",
    "get_drive",
    "upload_file_to_drive",
    "download_file_from_drive",
    "init_db_from_drive_once",
]