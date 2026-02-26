
# services/db.py
import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager

# 项目根目录：.../services/db.py 的上一级的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 允许用户/你在 Start.command 里覆盖路径（更灵活）
ENV_DB_PATH = os.getenv("MANLY_DB_PATH")

if ENV_DB_PATH:
    DB_PATH = ENV_DB_PATH
else:
    DB_DIR = PROJECT_ROOT / "db"   # 你也可以叫 data/
    DB_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH = str(DB_DIR / "manlyfarm.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def reset_db_connection():
    try:
        get_db.clear()
    except Exception:
        pass



@contextmanager
def db_connection():
    """数据库连接上下文管理器"""
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """初始化数据库表结构"""
    conn = get_db()
    cur = conn.cursor()

    # 创建 transactions 表
    cur.execute("""
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
            [Modifiers Applied] TEXT
        )
    """)

    # 创建 inventory 表
    cur.execute("""
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
        )
    """)

    # 创建 members 表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS members (
            [Square Customer ID] TEXT,
            [First Name] TEXT,
            [Last Name] TEXT,
            [Email Address] TEXT,
            [Phone Number] TEXT,
            [Creation Date] TEXT,
            [Customer Note] TEXT,
            [Reference ID] TEXT
        )
    """)

    # 创建 units 表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS units (
            name TEXT UNIQUE
        )
    """)

    # === NEW: ingestion log (track ingested source files) ===
    conn.execute("""
    CREATE TABLE IF NOT EXISTS ingestion_log (
        source_file TEXT PRIMARY KEY,
        ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # === NEW: high level summary table ===
    cur.execute("""
    CREATE TABLE IF NOT EXISTS high_level_daily (
        date TEXT,
        Category TEXT,
    
        daily_net_sales REAL,
        weekly_net_sales REAL,
        monthly_net_sales REAL,
    
        transactions INTEGER,
        customers INTEGER,
        qty REAL,
        avg_txn REAL,
    
        rolling_90 REAL,
        rolling_180 REAL,
    
        weekly_rolling_90 REAL,
        weekly_rolling_180 REAL,
    
        monthly_rolling_90 REAL,
        monthly_rolling_180 REAL,
    
        transactions_rolling_90 REAL,
        transactions_rolling_180 REAL,
    
        customers_rolling_90 REAL,
        customers_rolling_180 REAL,
    
        qty_rolling_90 REAL,
        qty_rolling_180 REAL,
    
        avg_txn_rolling_90 REAL,
        avg_txn_rolling_180 REAL,
    
        PRIMARY KEY (date, Category)
    )
    """)

    cur.execute('CREATE INDEX IF NOT EXISTS idx_high_level_date ON high_level_daily(date)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_high_level_category ON high_level_daily(Category)')


    # 创建索引以提高查询性能
    cur.execute('CREATE INDEX IF NOT EXISTS idx_txn_datetime ON transactions(Datetime)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_txn_id ON transactions([Transaction ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_member_square ON members([Square Customer ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_member_ref ON members([Reference ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_inv_sku ON inventory(SKU)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_inv_categories ON inventory(Categories)')

    conn.commit()
    conn.close()


def table_exists(table_name):
    """检查表是否存在"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cur.fetchone() is not None
    finally:
        conn.close()


def get_table_columns(table_name):
    """获取表的列信息"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def execute_query(query, params=None):
    """执行查询并返回结果"""
    conn = get_db()
    try:
        cur = conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)

        if query.strip().upper().startswith('SELECT'):
            return cur.fetchall()
        else:
            conn.commit()
            return cur.rowcount
    finally:
        conn.close()


def get_table_row_count(table_name):
    """获取表的行数"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()[0]
    except Exception:
        return 0
    finally:
        conn.close()

def get_db_path() -> str:
    """返回当前正式 DB 路径（用于临时库替换）。"""
    return DB_PATH


def get_db_with_path(path: str):
    """用指定路径打开 sqlite 连接（用于临时库）。"""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


# 导出函数
__all__ = [
    'get_db',
    'get_db_path',
    'get_db_with_path',
    'db_connection',
    'init_database',
    'table_exists',
    'get_table_columns',
    'execute_query',
    'get_table_row_count'
]
