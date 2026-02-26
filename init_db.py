# init_db.py
from services.db import init_database

def init_db():
    """初始化数据库（包装函数）"""
    init_database()
    print("✅ Database initialized successfully")

if __name__ == "__main__":
    init_db()