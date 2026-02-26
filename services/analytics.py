# services/analytics.py
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta
from services.db import get_db
import pandas as pd
from services.category_rules import is_bar_category

import math

def proper_round(x):
    if pd.isna(x):
        return 0
    x_rounded = round(float(x), 10)
    return math.floor(x_rounded + 0.5)

# === 工具函数 ===
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def _to_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )


# === 数据加载 ===
def load_transactions(db, days=365, time_from=None, time_to=None):
    if time_from and time_to:
        start, end = pd.to_datetime(time_from), pd.to_datetime(time_to)
    else:
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=days)

    # 转成 SQLite 可识别的字符串
    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end.strftime("%Y-%m-%d %H:%M:%S")

    query = """
        SELECT Datetime, Category, Item, Qty, [Net Sales], [Gross Sales],
               Discounts, [Customer ID], [Transaction ID]
        FROM transactions
        WHERE Datetime BETWEEN ? AND ?
    """
    df = pd.read_sql(query, db, params=[start_str, end_str])
    if not df.empty:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    return df


def load_inventory(db) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM inventory", db)
    return _clean_df(df)


def load_members(db) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM members", db)
    return _clean_df(df)


def compute_inventory_profit(df: pd.DataFrame) -> pd.DataFrame:
    """
    修改后的inventory value计算公式：
    - Tax - GST (10%)列如果是N: inventory value = Current Quantity Vie Market & Bar * Default Unit Cost
    - Tax - GST (10%)列如果是Y: inventory value = Current Quantity Vie Market & Bar * (Default Unit Cost/11*10)
    - 过滤掉Current Quantity Vie Market & Bar或者Default Unit Cost为空的行
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    for col in ["Tax - GST (10%)", "Price", "Current Quantity Vie Market & Bar", "Default Unit Cost"]:
        if col not in df.columns:
            df[col] = np.nan

    # 过滤掉空值行
    mask = (~df["Current Quantity Vie Market & Bar"].isna()) & (~df["Default Unit Cost"].isna())
    df = df[mask].copy()

    if df.empty:
        return df

    price = _to_numeric(df["Price"])
    qty = _to_numeric(df["Current Quantity Vie Market & Bar"])
    unit_cost = _to_numeric(df["Default Unit Cost"])
    tax_flag = df["Tax - GST (10%)"].astype(str)

    # 计算 retail_total
    retail_total = pd.Series(0.0, index=df.index)
    retail_total.loc[tax_flag.eq("N")] = (price * qty).loc[tax_flag.eq("N")]
    retail_total.loc[tax_flag.eq("Y")] = ((price / 11.0 * 10.0) * qty).loc[tax_flag.eq("Y")]

    # 修改：计算 inventory_value
    inventory_value = pd.Series(0.0, index=df.index)
    inventory_value.loc[tax_flag.eq("N")] = (unit_cost * qty).loc[tax_flag.eq("N")]
    inventory_value.loc[tax_flag.eq("Y")] = ((unit_cost / 11.0 * 10.0) * qty).loc[tax_flag.eq("Y")]

    profit = retail_total - inventory_value

    df["retail_total"] = retail_total
    df["inventory_value"] = inventory_value
    df["profit"] = profit

    return df


def load_all(db=None, time_from=None, time_to=None, days=None):
    conn = db or get_db()

    tx = pd.read_sql("SELECT * FROM transactions", conn)
    tx = tx.drop_duplicates(subset=[
        "Datetime",
        "Transaction ID",
        "Item",
        "Qty",
        "Net Sales"
    ])

    inv = pd.read_sql("SELECT * FROM inventory", conn)

    try:
        mem = pd.read_sql("SELECT * FROM members", conn)
    except Exception:
        mem = pd.DataFrame()

    # ✅ 每次都重新计算 inventory_value / profit，保证口径一致
    if not inv.empty:
        inv = compute_inventory_profit(inv)

    return tx, mem, inv


def daily_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()

    # 🔥 FIX 1: 统一日期
    transactions["date"] = pd.to_datetime(
        transactions["Datetime"], errors="coerce"
    ).dt.normalize()

    # 🔥 FIX 2: 确保金额字段为数值
    for col in ["Net Sales", "Gross Sales", "Qty"]:
        if col in transactions.columns:
            transactions[col] = (
                transactions[col]
                .astype(str)
                .str.replace(r"[^0-9\.\-]", "", regex=True)
                .replace("", pd.NA)
            )
            transactions[col] = pd.to_numeric(transactions[col], errors="coerce")

    # 🔥 FIX 3: 先按 date + Transaction ID 聚合（避免一单多商品重复）
    txn_level = (
        transactions.groupby(["date", "Transaction ID"], dropna=False)
        .agg(
            txn_net_sales=("Net Sales", "sum"),
            txn_gross_sales=("Gross Sales", "sum"),
            txn_qty=("Qty", "sum"),
            txn_customer=("Customer ID", "first"),
        )
        .reset_index()
    )

    # 🔥 FIX 4: 再按 date 汇总
    summary = (
        txn_level.groupby("date")
        .agg(
            net_sales=("txn_net_sales", "sum"),
            gross=("txn_gross_sales", "sum"),
            transactions=("Transaction ID", "nunique"),  # ✅ 正确交易数
            customers=("txn_customer", "nunique"),       # ✅ 正确客户数
            qty=("txn_qty", "sum"),
            avg_txn=("txn_net_sales", "mean"),           # ✅ 正确平均订单额
        )
        .reset_index()
    )

    summary["profit"] = summary["gross"] - summary["net_sales"]

    return summary


# === 销售预测 ===
def forecast_sales(transactions: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    transactions["date"] = pd.to_datetime(transactions["Datetime"]).dt.normalize()

    daily_sales = transactions.groupby("date")["Net Sales"].sum()
    if len(daily_sales) < 10:
        return pd.DataFrame()
    model = ExponentialSmoothing(daily_sales, trend="add", seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(periods)
    return pd.DataFrame({
        "date": pd.date_range(start=daily_sales.index[-1] + timedelta(days=1), periods=periods),
        "forecast": forecast.values
    })


# === 高消费客户 ===
def forecast_top_consumers(transactions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if transactions.empty or "Customer ID" not in transactions.columns:
        return pd.DataFrame()
    return (
        transactions.groupby("Customer ID")["Net Sales"]
        .sum()
        .reset_index()
        .sort_values("Net Sales", ascending=False)
        .head(top_n)
    )


# === SKU 消耗时序 ===
def sku_consumption_timeseries(transactions: pd.DataFrame, sku: str) -> pd.DataFrame:
    if transactions.empty or "Item" not in transactions.columns:
        return pd.DataFrame()
    df = transactions[transactions["Item"] == sku].copy()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["Datetime"]).dt.normalize()

    return df.groupby("date")["Qty"].sum().reset_index()


# === 会员相关分析 ===
def member_flagged_transactions(transactions: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """
    会员识别逻辑：
    1. 如果客户在会员表中，且交易日期 >= 会员开始日期 → enrolled
    2. 否则 → not enrolled
    3. 对于没有 First Visit 或 Creation Date 的客户，使用 First Visit 中的最大日期作为默认值
    """

    df = transactions.copy()

    # 没有 member 表 → 全部是非会员
    if members is None or members.empty:
        df["is_member"] = False
        return df

    # 确保 Datetime 是 datetime 类型
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    # 🔥 关键：使用 First Visit 中的最大日期作为默认值
    DEFAULT_MEMBER_DATE = None

    if "First Visit" in members.columns:
        # 提取所有有效的 First Visit 日期
        first_visit_dates = pd.to_datetime(members["First Visit"], errors="coerce").dropna()

        if not first_visit_dates.empty:
            # 使用 First Visit 中的最大日期作为默认值
            DEFAULT_MEMBER_DATE = first_visit_dates.max()
            print(f"DEBUG: 使用 First Visit 最大日期作为默认值: {DEFAULT_MEMBER_DATE.date()}")

    # 如果没有 First Visit，使用今天
    if DEFAULT_MEMBER_DATE is None:
        DEFAULT_MEMBER_DATE = pd.Timestamp.today().normalize()
        print(f"DEBUG: 没有 First Visit 数据，使用今天作为默认值: {DEFAULT_MEMBER_DATE.date()}")

    # === 为每个客户确定会员开始日期 ===
    customer_member_dates = {}

    # 处理 Square Customer ID
    if "Square Customer ID" in members.columns:
        for idx, row in members.iterrows():
            customer_id = str(row.get("Square Customer ID", "")).strip().lower()
            if not customer_id:
                continue

            # 确定会员开始日期
            member_start_date = None

            # 1. 优先使用 First Visit
            if "First Visit" in row:
                member_start_date = pd.to_datetime(row["First Visit"], errors="coerce")

            # 2. 其次使用 Creation Date
            if pd.isna(member_start_date) and "Creation Date" in row:
                member_start_date = pd.to_datetime(row["Creation Date"], errors="coerce")

            # 3. 🔥 关键修改：如果都没有，使用上面计算的默认日期
            if pd.isna(member_start_date):
                member_start_date = DEFAULT_MEMBER_DATE

            # 存储映射（如果有重复，取最早的日期）
            if customer_id not in customer_member_dates:
                customer_member_dates[customer_id] = member_start_date
            else:
                customer_member_dates[customer_id] = min(customer_member_dates[customer_id], member_start_date)

    # 处理 Reference ID
    if "Reference ID" in members.columns:
        for idx, row in members.iterrows():
            ref_id = str(row.get("Reference ID", "")).strip().lower()
            if not ref_id:
                continue

            # 确定会员开始日期（同上逻辑）
            member_start_date = None
            if "First Visit" in row:
                member_start_date = pd.to_datetime(row["First Visit"], errors="coerce")
            if pd.isna(member_start_date) and "Creation Date" in row:
                member_start_date = pd.to_datetime(row["Creation Date"], errors="coerce")

            # 🔥 使用默认日期
            if pd.isna(member_start_date):
                member_start_date = DEFAULT_MEMBER_DATE

            if ref_id not in customer_member_dates:
                customer_member_dates[ref_id] = member_start_date
            else:
                customer_member_dates[ref_id] = min(customer_member_dates[ref_id], member_start_date)

    # === 标记交易 ===
    df["clean_customer_id"] = df["Customer ID"].astype(str).str.strip().str.lower()
    df["is_member"] = False

    for idx, row in df.iterrows():
        customer_id = row.get("clean_customer_id", "")
        transaction_date = row["Datetime"]

        if not customer_id or pd.isna(transaction_date):
            continue

        if customer_id in customer_member_dates:
            member_start_date = customer_member_dates[customer_id]

            # 交易日期在会员开始日期之后（或当天）才算是会员消费
            if pd.notna(member_start_date) and transaction_date >= member_start_date:
                df.at[idx, "is_member"] = True

    # 清理临时列
    df = df.drop(columns=["clean_customer_id"], errors="ignore")

    # 添加调试信息
    print(f"DEBUG: 总客户数: {len(customer_member_dates)}")

    # 统计使用默认日期的客户数
    default_date_count = 0
    specific_date_count = 0

    for customer_id, date in customer_member_dates.items():
        if pd.notna(date):
            if date == DEFAULT_MEMBER_DATE:
                default_date_count += 1
            else:
                specific_date_count += 1

    print(f"DEBUG: 使用特定日期的客户数: {specific_date_count}")
    print(f"DEBUG: 使用默认日期的客户数: {default_date_count}")
    print(f"DEBUG: 默认日期值: {DEFAULT_MEMBER_DATE.date()}")
    print(f"DEBUG: 会员交易数: {df['is_member'].sum()}")
    print(f"DEBUG: 非会员交易数: {(df['is_member'] == False).sum()}")

    # 检查一些示例客户的会员开始日期
    if customer_member_dates:
        sample_customers = list(customer_member_dates.items())[:3]
        print(f"DEBUG: 示例客户会员开始日期:")
        for customer_id, date in sample_customers:
            date_str = date.date() if pd.notna(date) else "NaN"
            print(f"  - {customer_id}: {date_str}")

    return df

def member_frequency_stats(transactions: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or members.empty:
        return pd.DataFrame()
    df = transactions.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    stats = (
        df.groupby("Customer ID")["Datetime"]
        .agg(["count", "min", "max"])
        .reset_index()
        .rename(columns={"count": "txn_count", "min": "first_txn", "max": "last_txn"})
    )
    # ✅ 用新的列名来计算
    stats["days_active"] = (stats["last_txn"] - stats["first_txn"]).dt.days.clip(lower=1)
    stats["avg_days_between"] = stats["days_active"] / stats["txn_count"]
    return stats


def non_member_overview(transactions: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    member_ids = set(members["Square Customer ID"].unique()) if not members.empty else set()
    df = transactions[~transactions["Customer ID"].isin(member_ids)].copy()
    return df.groupby("Customer ID")["Net Sales"].sum().reset_index()


# === 分类与推荐分析 ===
def category_counts(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "Category" not in transactions.columns:
        return pd.DataFrame()
    return transactions["Category"].value_counts().reset_index().rename(
        columns={"index": "Category", "Category": "count"})


def heatmap_pivot(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "Category" not in transactions.columns:
        return pd.DataFrame()
    return pd.pivot_table(
        transactions, values="Net Sales", index="Customer ID", columns="Category", aggfunc="sum", fill_value=0
    )


def top_categories_for_customer(transactions: pd.DataFrame, customer_id: str, top_n: int = 3) -> pd.DataFrame:
    df = transactions[transactions["Customer ID"] == customer_id]
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("Category")["Net Sales"]
        .sum()
        .reset_index()
        .sort_values("Net Sales", ascending=False)
        .head(top_n)
    )


def recommend_similar_categories(transactions: pd.DataFrame, category: str, top_n: int = 3) -> pd.DataFrame:
    if transactions.empty or "Category" not in transactions.columns:
        return pd.DataFrame()
    other_cats = transactions["Category"].value_counts().reset_index()
    other_cats = other_cats[other_cats["index"] != category]
    return other_cats.head(top_n)


def ltv_timeseries_for_customer(transactions: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    df = transactions[transactions["Customer ID"] == customer_id]
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["Datetime"]).dt.normalize()

    return df.groupby("date")["Net Sales"].sum().cumsum().reset_index()


def recommend_bundles_for_customer(transactions: pd.DataFrame, customer_id: str, top_n: int = 3) -> pd.DataFrame:
    df = transactions[transactions["Customer ID"] == customer_id]
    if df.empty or "Item" not in df.columns:
        return pd.DataFrame()
    return df["Item"].value_counts().reset_index().head(top_n)


def churn_signals_for_member(transactions: pd.DataFrame, members: pd.DataFrame,
                             days_threshold: int = 30) -> pd.DataFrame:
    if transactions.empty or members.empty:
        return pd.DataFrame()
    df = transactions[transactions["Customer ID"].isin(members["Square Customer ID"].unique())]
    if df.empty:
        return pd.DataFrame()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    last_seen = df.groupby("Customer ID")["Datetime"].max().reset_index()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_threshold)
    last_seen["churn_flag"] = last_seen["Datetime"] < cutoff
    return last_seen

def rebuild_inventory_summary():
    """
    基于当前 inventory 表重新构建 inventory_summary。
    使用 v27 逻辑。
    - 完全数据库驱动
    - 覆盖式重建
    - 列名统一使用 Product Name
    """

    import pandas as pd
    import numpy as np
    from services.db import get_db, db_connection
    from services.category_rules import is_bar_category

    conn = get_db()

    # 🔥 强制重建表结构
    conn.execute("DROP TABLE IF EXISTS inventory_summary")
    conn.commit()

    conn.execute("""
    CREATE TABLE inventory_summary (
        date TEXT,
        Category TEXT,
        inventory_value REAL,
        profit REAL,
        inventory_rolling_90 REAL,
        inventory_rolling_180 REAL,
        PRIMARY KEY (date, Category)
    )
    """)

    # 🔥 创建索引（必须在建表之后）
    conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_summary_date ON inventory_summary(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_inv_summary_category ON inventory_summary(Category)")
    
    conn.commit()

    # ========= 1️⃣ 读取 inventory =========
    inv = pd.read_sql("SELECT * FROM inventory", conn)

    if inv.empty:
        conn.execute("DELETE FROM inventory_summary")
        conn.commit()
        conn.close()
        return

    df = inv.copy()

    # ========= 2️⃣ 日期处理 =========
    df["date"] = pd.to_datetime(df["source_date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # ========= 3️⃣ 商品名称（使用 Product Name） =========
    if "Product Name" in df.columns:
        df["Product Name"] = df["Product Name"].astype(str).str.strip()
    else:
        df["Product Name"] = "Unknown"

    # ========= 4️⃣ Category =========
    if "Categories" in df.columns:
        df["Category"] = df["Categories"].astype(str)
    elif "Category" in df.columns:
        df["Category"] = df["Category"].astype(str)
    else:
        df["Category"] = "Unknown"

    # ========= 5️⃣ 数值处理 =========
    df["Quantity"] = pd.to_numeric(
        df.get("Current Quantity Vie Market & Bar", 0),
        errors="coerce"
    )

    df = df[df["Quantity"] > 0].copy()

    df["UnitCost"] = pd.to_numeric(
        df.get("Default Unit Cost", 0),
        errors="coerce"
    ).fillna(0)

    df["Price"] = pd.to_numeric(
        df.get("Price", 0),
        errors="coerce"
    ).fillna(0)

    tax_col = "Tax - GST (10%)"
    if tax_col not in df.columns:
        df[tax_col] = "N"

    df[tax_col] = df[tax_col].astype(str).str.upper()

    # ========= 6️⃣ inventory_value =========
    def calc_inventory(row):
        if row[tax_col] == "Y":
            return (row["UnitCost"] / 11 * 10) * row["Quantity"]
        return row["UnitCost"] * row["Quantity"]

    df["inventory_value"] = df.apply(calc_inventory, axis=1)

    # ========= 7️⃣ retail_total =========
    def calc_retail(row):
        if row[tax_col] == "Y":
            return (row["Price"] / 11 * 10) * row["Quantity"]
        return row["Price"] * row["Quantity"]

    df["retail_total"] = df.apply(calc_retail, axis=1)

    df["profit"] = df["retail_total"] - df["inventory_value"]

    # ========= 8️⃣ 按 date + Category 聚合 =========
    g = (
        df.groupby(["date", "Category"], as_index=False)[
            ["inventory_value", "profit"]
        ].sum()
    )
    g = g.sort_values(["Category", "date"])

    g["inventory_rolling_90"] = (
        g.groupby("Category")["inventory_value"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    g["inventory_rolling_180"] = (
        g.groupby("Category")["inventory_value"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ========= 9️⃣ bar / retail / total =========
    g["is_bar"] = g["Category"].apply(is_bar_category)

    def build_group(df_part, label):
        if df_part.empty:
            return pd.DataFrame()

        out = (
            df_part.groupby("date", as_index=False)[
                ["inventory_value", "profit"]
            ].sum()
        )

        out["Category"] = label
        out = out.sort_values("date")

        # 🔥 关键：重新计算 rolling
        out["inventory_rolling_90"] = (
            out["inventory_value"]
            .rolling(90, min_periods=1)
            .mean()
        )

        out["inventory_rolling_180"] = (
            out["inventory_value"]
            .rolling(180, min_periods=1)
            .mean()
        )

        return out

    bar = build_group(g[g["is_bar"]], "bar")
    retail = build_group(g[~g["is_bar"]], "retail")
    total = build_group(g, "total")

    final = pd.concat(
        [g.drop(columns=["is_bar"]), bar, retail, total],
        ignore_index=True
    )

    # ========= 🔟 覆盖式更新 =========
    conn.execute("DELETE FROM inventory_summary")
    conn.commit()
    conn.close()

    with db_connection() as conn:
        for _, row in final.iterrows():
            conn.execute("""
                INSERT INTO inventory_summary (
                    date,
                    Category,
                    inventory_value,
                    profit,
                    inventory_rolling_90,
                    inventory_rolling_180
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                    row["date"],
                    row["Category"],
                    float(row["inventory_value"]),
                    float(row["profit"]),
                    float(row["inventory_rolling_90"]),
                    float(row["inventory_rolling_180"]),
            ))

        conn.commit()

def rebuild_high_level_summary():
    """
    使用 27 版本 high level 原始计算逻辑
    全量重建 high_level_daily
    """

    import pandas as pd
    import numpy as np
    from services.db import get_db
    from services.category_rules import is_bar_category

    conn = get_db()

    conn.execute("DROP TABLE IF EXISTS high_level_daily")
    conn.commit()

    # =========================
    # ① 读取 transactions
    # =========================
    tx = pd.read_sql("SELECT * FROM transactions", conn)
    # 🔥 完全复刻 27 版本去重逻辑
    tx = tx.drop_duplicates(subset=[
        "Datetime",
        "Transaction ID",
        "Item",
        "Qty",
        "Net Sales"
    ])

    if tx.empty:
        conn.close()
        return

    tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")
    tx = tx[tx["Datetime"].notna()].copy()

    tx["date"] = tx["Datetime"].dt.normalize()

    tx["Category"] = (
        tx["Category"]
        .fillna("None")
        .astype(str)
        .str.strip()
        .replace("", "None")
    )

    # ==============================
    # 🔥🔥🔥 在这里插入 27 版本 customers 口径代码
    # ==============================

    # ====== 27 版本 customers 口径：用 Card Brand + PAN Suffix ======
    tx_customers = tx.copy()

    tx_customers = tx_customers.dropna(
        subset=["date", "Card Brand", "PAN Suffix"]
    )

    tx_customers["Card Brand"] = (
        tx_customers["Card Brand"]
        .astype(str)
        .str.title()
    )

    tx_customers["PAN Suffix"] = (
        tx_customers["PAN Suffix"]
        .astype(str)
        .str.split(".")
        .str[0]
    )

    # total customers
    unique_pairs = tx_customers[
        ["date", "Card Brand", "PAN Suffix"]
    ].drop_duplicates()

    cust_total = (
        unique_pairs.groupby("date")
        .size()
        .reset_index(name="customers")
    )

    # customers by category
    unique_pairs_cat = tx_customers[
        ["date", "Category", "Card Brand", "PAN Suffix"]
    ].drop_duplicates()

    cust_by_cat = (
        unique_pairs_cat.groupby(["date", "Category"])
        .size()
        .reset_index(name="customers")
    )

    unique_pairs_cat["__is_bar__"] = unique_pairs_cat["Category"].apply(is_bar_category)

    cust_bar = (
        unique_pairs_cat[unique_pairs_cat["__is_bar__"]]
        .groupby("date")
        .size()
        .reset_index(name="customers")
    )

    cust_retail = (
        unique_pairs_cat[~unique_pairs_cat["__is_bar__"]]
        .groupby("date")
        .size()
        .reset_index(name="customers")
    )

    # =========================
    # ② 先按 交易ID 聚合（和 27 一模一样）
    # =========================
    txn_level = (
        tx.groupby(["date", "Transaction ID", "Category"])
        .agg(
            txn_net_sales=("Net Sales", "sum"),
            txn_qty=("Qty", "sum"),
        )
        .reset_index()
    )

    # =========================
    # ③ 再按 date + Category 聚合
    # =========================
    base = (
        txn_level.groupby(["date", "Category"])
        .agg(
            daily_net_sales=("txn_net_sales", "sum"),
            transactions=("Transaction ID", "nunique"),
            qty=("txn_qty", "sum"),
        )
        .reset_index()
    )

    base = base.merge(
        cust_by_cat.rename(columns={"date": "date"}),
        on=["date", "Category"],
        how="left"
    )

    base["customers"] = base["customers"].fillna(0)

    base["avg_txn"] = np.where(
        base["transactions"] > 0,
        base["daily_net_sales"] / base["transactions"],
        0
    )

    # =========================
    # ④ Weekly / Monthly
    # =========================
    base["week_start"] = base["date"] - pd.to_timedelta(base["date"].dt.weekday, unit="D")
    base["weekly_net_sales"] = (
        base.groupby(["week_start", "Category"])["daily_net_sales"]
        .transform("sum")
    )
    base = base.drop(columns=["week_start"])

    base["month_start"] = base["date"].values.astype("datetime64[M]")
    base["monthly_net_sales"] = (
        base.groupby(["month_start", "Category"])["daily_net_sales"]
        .transform("sum")
    )
    base = base.drop(columns=["month_start"])

    # =========================
    # ⑤ 补齐日期（27 版本关键）
    # =========================
    all_dates = pd.date_range(
        base["date"].min(),
        base["date"].max(),
        freq="D"
    )

    fixed = []

    for cat, g in base.groupby("Category"):
        g = g.set_index("date").sort_index()
        g = g.reindex(all_dates)
        g["Category"] = cat

        for col in [
            "daily_net_sales",
            "weekly_net_sales",
            "monthly_net_sales",
            "transactions",
            "customers",
            "qty",
            "avg_txn"
        ]:
            g[col] = g[col].fillna(0)

        g = g.reset_index().rename(columns={"index": "date"})
        fixed.append(g)

    base = pd.concat(fixed, ignore_index=True)
    base = base.sort_values(["Category", "date"])

    # =========================
    # ⑥ Rolling（和 27 一样）
    # =========================
    base["rolling_90"] = (
        base.groupby("Category")["daily_net_sales"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["rolling_180"] = (
        base.groupby("Category")["daily_net_sales"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["weekly_rolling_90"] = (
        base.groupby("Category")["weekly_net_sales"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["weekly_rolling_180"] = (
        base.groupby("Category")["weekly_net_sales"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["monthly_rolling_90"] = (
        base.groupby("Category")["monthly_net_sales"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["monthly_rolling_180"] = (
        base.groupby("Category")["monthly_net_sales"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ===== Transactions Rolling =====
    base["transactions_rolling_90"] = (
        base.groupby("Category")["transactions"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["transactions_rolling_180"] = (
        base.groupby("Category")["transactions"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ===== Customers Rolling =====
    base["customers_rolling_90"] = (
        base.groupby("Category")["customers"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["customers_rolling_180"] = (
        base.groupby("Category")["customers"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ===== Qty Rolling =====
    base["qty_rolling_90"] = (
        base.groupby("Category")["qty"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["qty_rolling_180"] = (
        base.groupby("Category")["qty"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ===== Avg Transaction Rolling =====
    base["avg_txn_rolling_90"] = (
        base.groupby("Category")["avg_txn"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["avg_txn_rolling_180"] = (
        base.groupby("Category")["avg_txn"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    # =========================
    # ⑦ Bar / Retail / Total  ✅ 修复：不要从 base(sum) 得到
    #    要从 txn_level 回算 transactions/customers/avg/rolling
    # =========================

    def _build_group_from_txn_level(txn_level_df: pd.DataFrame, label: str, customers_df: pd.DataFrame) -> pd.DataFrame:

        if txn_level_df.empty:
            g = pd.DataFrame({"date": all_dates})
            for c in [
                "daily_net_sales", "weekly_net_sales", "monthly_net_sales",
                "transactions", "customers", "qty", "avg_txn",
                "rolling_90", "rolling_180",
                "weekly_rolling_90", "weekly_rolling_180",
                "monthly_rolling_90", "monthly_rolling_180",
            ]:
                g[c] = 0.0
            g["Category"] = label
            return g

        # ========= 1️⃣ Daily 聚合 =========
        g = (
            txn_level_df.groupby("date", as_index=False)
            .agg(
                daily_net_sales=("txn_net_sales", "sum"),
                transactions=("Transaction ID", "nunique"),
                qty=("txn_qty", "sum"),
            )
        )

        g["avg_txn"] = np.where(
            g["transactions"] > 0,
            g["daily_net_sales"] / g["transactions"],
            0
        )

        # ========= 2️⃣ Weekly / Monthly =========
        g["week_start"] = g["date"] - pd.to_timedelta(g["date"].dt.weekday, unit="D")
        g["weekly_net_sales"] = g.groupby("week_start")["daily_net_sales"].transform("sum")
        g = g.drop(columns=["week_start"])

        g["month_start"] = g["date"].values.astype("datetime64[M]")
        g["monthly_net_sales"] = g.groupby("month_start")["daily_net_sales"].transform("sum")
        g = g.drop(columns=["month_start"])

        # ========= 3️⃣ 补齐日期 =========
        g = g.set_index("date").sort_index().reindex(all_dates)
        g["Category"] = label

        for col in [
            "daily_net_sales", "weekly_net_sales", "monthly_net_sales",
            "transactions", "qty", "avg_txn"
        ]:
            g[col] = g[col].fillna(0)

        g = g.reset_index().rename(columns={"index": "date"})

        # ========= 4️⃣ merge customers（关键） =========
        customers_df = customers_df.copy()
        customers_df["date"] = pd.to_datetime(customers_df["date"], errors="coerce").dt.normalize()

        g = g.merge(customers_df[["date", "customers"]], on="date", how="left")
        g["customers"] = g["customers"].fillna(0)

        # ========= 5️⃣ Rolling =========
        g = g.sort_values("date")

        g["rolling_90"] = g["daily_net_sales"].rolling(90, min_periods=1).mean()
        g["rolling_180"] = g["daily_net_sales"].rolling(180, min_periods=1).mean()

        g["weekly_rolling_90"] = g["weekly_net_sales"].rolling(90, min_periods=1).mean()
        g["weekly_rolling_180"] = g["weekly_net_sales"].rolling(180, min_periods=1).mean()

        g["monthly_rolling_90"] = g["monthly_net_sales"].rolling(90, min_periods=1).mean()
        g["monthly_rolling_180"] = g["monthly_net_sales"].rolling(180, min_periods=1).mean()

        g["transactions_rolling_90"] = g["transactions"].rolling(90, min_periods=1).mean()
        g["transactions_rolling_180"] = g["transactions"].rolling(180, min_periods=1).mean()

        g["customers_rolling_90"] = g["customers"].rolling(90, min_periods=1).mean()
        g["customers_rolling_180"] = g["customers"].rolling(180, min_periods=1).mean()

        g["qty_rolling_90"] = g["qty"].rolling(90, min_periods=1).mean()
        g["qty_rolling_180"] = g["qty"].rolling(180, min_periods=1).mean()

        g["avg_txn_rolling_90"] = g["avg_txn"].rolling(90, min_periods=1).mean()
        g["avg_txn_rolling_180"] = g["avg_txn"].rolling(180, min_periods=1).mean()

        return g

    # ✅ 用 txn_level 来切 bar/retail/total（不是用 base 去 sum）
    txn_level["is_bar"] = txn_level["Category"].apply(is_bar_category)

    txn_bar = txn_level[txn_level["is_bar"]].copy()
    txn_retail = txn_level[~txn_level["is_bar"]].copy()

    total = _build_group_from_txn_level(txn_level, "total", cust_total.rename(columns={"date": "date"}))
    # =========================
    # 27 VERSION: customers 分摊
    # =========================

    # 1️⃣ 计算每日交易数
    tx_total = (
        txn_level.groupby("date")["Transaction ID"]
        .nunique()
        .reset_index(name="tx_total")
    )

    tx_bar = (
        txn_bar.groupby("date")["Transaction ID"]
        .nunique()
        .reset_index(name="tx_bar")
    )

    # 2️⃣ 合并 total customers
    ratio_df = (
        cust_total.merge(tx_total, on="date", how="left")
        .merge(tx_bar, on="date", how="left")
        .fillna(0)
    )

    # 3️⃣ 按交易比例分摊
    ratio_df["bar_customers"] = np.where(
        ratio_df["tx_total"] > 0,
        (ratio_df["customers"] * ratio_df["tx_bar"] / ratio_df["tx_total"]).astype(int),
        0
    )

    ratio_df["retail_customers"] = (
            ratio_df["customers"] - ratio_df["bar_customers"]
    )

    # 4️⃣ 构造新的 customers dataframe
    cust_bar_alloc = ratio_df[["date", "bar_customers"]].rename(
        columns={"bar_customers": "customers"}
    )

    cust_retail_alloc = ratio_df[["date", "retail_customers"]].rename(
        columns={"retail_customers": "customers"}
    )

    bar = _build_group_from_txn_level(txn_bar, "bar", cust_bar_alloc)
    retail = _build_group_from_txn_level(txn_retail, "retail", cust_retail_alloc)

    # base 里不需要 is_bar 这列（避免写入 DB 多余列）
    base = base.drop(columns=["is_bar"], errors="ignore")

    final = pd.concat([base, bar, retail, total], ignore_index=True)

    final = final[[
        "date", "Category",
        "daily_net_sales",
        "weekly_net_sales",
        "monthly_net_sales",
        "transactions",
        "customers",
        "qty",
        "avg_txn",
        "rolling_90",
        "rolling_180",
        "weekly_rolling_90",
        "weekly_rolling_180",
        "monthly_rolling_90",
        "monthly_rolling_180",
        "transactions_rolling_90",
        "transactions_rolling_180",
        "customers_rolling_90",
        "customers_rolling_180",
        "qty_rolling_90",
        "qty_rolling_180",
        "avg_txn_rolling_90",
        "avg_txn_rolling_180",
    ]]

    final["date"] = pd.to_datetime(final["date"]).dt.strftime("%Y-%m-%d")

    final.to_sql("high_level_daily", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

def update_high_level_summary_by_db_diff():
    """
    基于数据库差异同步 high_level_daily（用于 Refresh New Files / Drive 同步后）
    """

    import pandas as pd
    import numpy as np
    from services.db import get_db
    from services.category_rules import is_bar_category

    conn = get_db()

    # 1) transactions 的日期全集
    tx_dates_df = pd.read_sql(
        "SELECT DISTINCT date(Datetime) AS d FROM transactions",
        conn
    )
    tx_set = set(tx_dates_df["d"].dropna().astype(str).tolist())

    # 2) summary 的日期全集
    summary_dates_df = pd.read_sql(
        "SELECT DISTINCT date AS d FROM high_level_daily",
        conn
    )
    summary_set = set(summary_dates_df["d"].dropna().astype(str).tolist())

    # 如果 transactions 为空：清空 summary
    if not tx_set:
        conn.execute("DELETE FROM high_level_daily")
        conn.commit()
        conn.close()
        return

    # 3) summary 多出来的日期 -> 删除
    deleted_dates = summary_set - tx_set
    for d in deleted_dates:
        conn.execute("DELETE FROM high_level_daily WHERE date = ?", (d,))

    # 4) transactions 新增/缺失的日期
    missing_dates = tx_set - summary_set

    if not missing_dates and not deleted_dates:
        conn.commit()
        conn.close()
        return

    # 5) rolling 安全窗口
    affected_dates = sorted(list(missing_dates or deleted_dates))
    min_affected = pd.to_datetime(affected_dates[0]).normalize()
    recalc_start = (
            pd.to_datetime(min(tx_set)) - pd.Timedelta(days=180)
    ).normalize()
    recalc_start_str = recalc_start.strftime("%Y-%m-%d")

    # 删除窗口内旧数据
    conn.execute("DELETE FROM high_level_daily WHERE date >= ?", (recalc_start_str,))
    conn.commit()

    # 6) 读取窗口内交易数据
    tx = pd.read_sql(
        "SELECT * FROM transactions WHERE date(Datetime) >= ?",
        conn,
        params=[recalc_start_str],
    )

    if tx.empty:
        conn.close()
        return

    tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")
    tx = tx[tx["Datetime"].notna()].copy()

    tx["date"] = tx["Datetime"].dt.normalize()

    tx["Category"] = (
        tx["Category"]
        .fillna("None")
        .astype(str)
        .str.strip()
        .replace("", "None")
    )

    # ==============================
    # 🔥🔥🔥 在这里插入 27 版本 customers 口径代码
    # ==============================

    # ====== 27 版本 customers 口径：用 Card Brand + PAN Suffix ======
    tx_customers = tx.copy()

    tx_customers = tx_customers.dropna(
        subset=["date", "Card Brand", "PAN Suffix"]
    )

    tx_customers["Card Brand"] = (
        tx_customers["Card Brand"]
        .astype(str)
        .str.title()
    )

    tx_customers["PAN Suffix"] = (
        tx_customers["PAN Suffix"]
        .astype(str)
        .str.split(".")
        .str[0]
    )

    # total customers
    unique_pairs = tx_customers[
        ["date", "Card Brand", "PAN Suffix"]
    ].drop_duplicates()

    cust_total = (
        unique_pairs.groupby("date")
        .size()
        .reset_index(name="customers")
    )

    # customers by category
    unique_pairs_cat = tx_customers[
        ["date", "Category", "Card Brand", "PAN Suffix"]
    ].drop_duplicates()

    cust_by_cat = (
        unique_pairs_cat.groupby(["date", "Category"])
        .size()
        .reset_index(name="customers")
    )

    unique_pairs_cat["__is_bar__"] = unique_pairs_cat["Category"].apply(is_bar_category)

    cust_bar = (
        unique_pairs_cat[unique_pairs_cat["__is_bar__"]]
        .groupby("date")
        .size()
        .reset_index(name="customers")
    )

    cust_retail = (
        unique_pairs_cat[~unique_pairs_cat["__is_bar__"]]
        .groupby("date")
        .size()
        .reset_index(name="customers")
    )

    # =========================
    # ① 交易级聚合（和 rebuild 完全一致）
    # =========================
    txn_level = (
        tx.groupby(["date", "Transaction ID", "Category"])
        .agg(
            txn_net_sales=("Net Sales", "sum"),
            txn_qty=("Qty", "sum"),
        )
        .reset_index()
    )

    base = (
        txn_level.groupby(["date", "Category"])
        .agg(
            daily_net_sales=("txn_net_sales", "sum"),
            transactions=("Transaction ID", "nunique"),
            qty=("txn_qty", "sum"),
        )
        .reset_index()
    )

    base = base.merge(
        cust_by_cat.rename(columns={"date": "date"}),
        on=["date", "Category"],
        how="left"
    )

    base["customers"] = base["customers"].fillna(0)


    base["avg_txn"] = np.where(
        base["transactions"] > 0,
        base["daily_net_sales"] / base["transactions"],
        0
    )

    # =========================
    # Weekly / Monthly
    # =========================
    base["week_start"] = base["date"] - pd.to_timedelta(base["date"].dt.weekday, unit="D")
    base["weekly_net_sales"] = (
        base.groupby(["week_start", "Category"])["daily_net_sales"]
        .transform("sum")
    )
    base = base.drop(columns=["week_start"])

    base["month_start"] = base["date"].values.astype("datetime64[M]")
    base["monthly_net_sales"] = (
        base.groupby(["month_start", "Category"])["daily_net_sales"]
        .transform("sum")
    )
    base = base.drop(columns=["month_start"])

    # =========================
    # 补齐日期（关键）
    # =========================
    all_dates = pd.date_range(
        base["date"].min(),
        base["date"].max(),
        freq="D"
    )

    fixed = []

    for cat, g in base.groupby("Category"):
        g = g.set_index("date").sort_index()
        g = g.reindex(all_dates)
        g["Category"] = cat

        for col in [
            "daily_net_sales",
            "weekly_net_sales",
            "monthly_net_sales",
            "transactions",
            "customers",
            "qty",
            "avg_txn"
        ]:
            g[col] = g[col].fillna(0)

        g = g.reset_index().rename(columns={"index": "date"})
        fixed.append(g)

    base = pd.concat(fixed, ignore_index=True)
    base = base.sort_values(["Category", "date"])

    # =========================
    # Rolling
    # =========================
    base["rolling_90"] = (
        base.groupby("Category")["daily_net_sales"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["rolling_180"] = (
        base.groupby("Category")["daily_net_sales"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["weekly_rolling_90"] = (
        base.groupby("Category")["weekly_net_sales"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["weekly_rolling_180"] = (
        base.groupby("Category")["weekly_net_sales"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["monthly_rolling_90"] = (
        base.groupby("Category")["monthly_net_sales"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["monthly_rolling_180"] = (
        base.groupby("Category")["monthly_net_sales"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    # ===== Transactions Rolling =====
    base["transactions_rolling_90"] = (
        base.groupby("Category")["transactions"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["transactions_rolling_180"] = (
        base.groupby("Category")["transactions"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ===== Customers Rolling =====
    base["customers_rolling_90"] = (
        base.groupby("Category")["customers"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["customers_rolling_180"] = (
        base.groupby("Category")["customers"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ===== Qty Rolling =====
    base["qty_rolling_90"] = (
        base.groupby("Category")["qty"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["qty_rolling_180"] = (
        base.groupby("Category")["qty"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ===== Avg Transaction Rolling =====
    base["avg_txn_rolling_90"] = (
        base.groupby("Category")["avg_txn"]
        .rolling(90, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    base["avg_txn_rolling_180"] = (
        base.groupby("Category")["avg_txn"]
        .rolling(180, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    def _build_group_from_txn_level(txn_level_df: pd.DataFrame, label: str, customers_df: pd.DataFrame) -> pd.DataFrame:

        if txn_level_df.empty:
            g = pd.DataFrame({"date": all_dates})
            for c in [
                "daily_net_sales", "weekly_net_sales", "monthly_net_sales",
                "transactions", "customers", "qty", "avg_txn",
                "rolling_90", "rolling_180",
                "weekly_rolling_90", "weekly_rolling_180",
                "monthly_rolling_90", "monthly_rolling_180",
            ]:
                g[c] = 0.0
            g["Category"] = label
            return g

        # ========= 1️⃣ Daily 聚合 =========
        g = (
            txn_level_df.groupby("date", as_index=False)
            .agg(
                daily_net_sales=("txn_net_sales", "sum"),
                transactions=("Transaction ID", "nunique"),
                qty=("txn_qty", "sum"),
            )
        )

        g["avg_txn"] = np.where(
            g["transactions"] > 0,
            g["daily_net_sales"] / g["transactions"],
            0
        )

        # ========= 2️⃣ Weekly / Monthly =========
        g["week_start"] = g["date"] - pd.to_timedelta(g["date"].dt.weekday, unit="D")
        g["weekly_net_sales"] = g.groupby("week_start")["daily_net_sales"].transform("sum")
        g = g.drop(columns=["week_start"])

        g["month_start"] = g["date"].values.astype("datetime64[M]")
        g["monthly_net_sales"] = g.groupby("month_start")["daily_net_sales"].transform("sum")
        g = g.drop(columns=["month_start"])

        # ========= 3️⃣ 补齐日期 =========
        g = g.set_index("date").sort_index().reindex(all_dates)
        g["Category"] = label

        for col in [
            "daily_net_sales", "weekly_net_sales", "monthly_net_sales",
            "transactions", "qty", "avg_txn"
        ]:
            g[col] = g[col].fillna(0)

        g = g.reset_index().rename(columns={"index": "date"})

        # ========= 4️⃣ merge customers（关键） =========
        customers_df = customers_df.copy()
        customers_df["date"] = pd.to_datetime(customers_df["date"], errors="coerce").dt.normalize()

        g = g.merge(customers_df[["date", "customers"]], on="date", how="left")
        g["customers"] = g["customers"].fillna(0)

        # ========= 5️⃣ Rolling =========
        g = g.sort_values("date")

        g["rolling_90"] = g["daily_net_sales"].rolling(90, min_periods=1).mean()
        g["rolling_180"] = g["daily_net_sales"].rolling(180, min_periods=1).mean()

        g["weekly_rolling_90"] = g["weekly_net_sales"].rolling(90, min_periods=1).mean()
        g["weekly_rolling_180"] = g["weekly_net_sales"].rolling(180, min_periods=1).mean()

        g["monthly_rolling_90"] = g["monthly_net_sales"].rolling(90, min_periods=1).mean()
        g["monthly_rolling_180"] = g["monthly_net_sales"].rolling(180, min_periods=1).mean()

        g["transactions_rolling_90"] = g["transactions"].rolling(90, min_periods=1).mean()
        g["transactions_rolling_180"] = g["transactions"].rolling(180, min_periods=1).mean()

        g["customers_rolling_90"] = g["customers"].rolling(90, min_periods=1).mean()
        g["customers_rolling_180"] = g["customers"].rolling(180, min_periods=1).mean()

        g["qty_rolling_90"] = g["qty"].rolling(90, min_periods=1).mean()
        g["qty_rolling_180"] = g["qty"].rolling(180, min_periods=1).mean()

        g["avg_txn_rolling_90"] = g["avg_txn"].rolling(90, min_periods=1).mean()
        g["avg_txn_rolling_180"] = g["avg_txn"].rolling(180, min_periods=1).mean()

        return g
    # ✅ 用 txn_level 来切 bar/retail/total（不是用 base 去 sum）
    txn_level["is_bar"] = txn_level["Category"].apply(is_bar_category)

    txn_bar = txn_level[txn_level["is_bar"]].copy()
    txn_retail = txn_level[~txn_level["is_bar"]].copy()

    total = _build_group_from_txn_level(
        txn_level,
        "total",
        cust_total
    )
    # =========================
    # 27 VERSION: customers 分摊
    # =========================

    # 1️⃣ 计算每日交易数
    tx_total = (
        txn_level.groupby("date")["Transaction ID"]
        .nunique()
        .reset_index(name="tx_total")
    )

    tx_bar = (
        txn_bar.groupby("date")["Transaction ID"]
        .nunique()
        .reset_index(name="tx_bar")
    )

    # 2️⃣ 合并 total customers
    ratio_df = (
        cust_total.merge(tx_total, on="date", how="left")
        .merge(tx_bar, on="date", how="left")
        .fillna(0)
    )

    # 3️⃣ 按交易比例分摊
    ratio_df["bar_customers"] = np.where(
        ratio_df["tx_total"] > 0,
        (ratio_df["customers"] * ratio_df["tx_bar"] / ratio_df["tx_total"]).astype(int),
        0
    )

    ratio_df["retail_customers"] = (
            ratio_df["customers"] - ratio_df["bar_customers"]
    )

    # 4️⃣ 构造新的 customers dataframe
    cust_bar_alloc = ratio_df[["date", "bar_customers"]].rename(
        columns={"bar_customers": "customers"}
    )

    cust_retail_alloc = ratio_df[["date", "retail_customers"]].rename(
        columns={"retail_customers": "customers"}
    )
    bar = _build_group_from_txn_level(txn_bar, "bar", cust_bar_alloc)
    retail = _build_group_from_txn_level(txn_retail, "retail", cust_retail_alloc)
    # base 里不需要 is_bar 这列（避免写入 DB 多余列）
    base = base.drop(columns=["is_bar"], errors="ignore")

    final = pd.concat([base, bar, retail, total], ignore_index=True)

    final = final[[
        "date", "Category",
        "daily_net_sales",
        "weekly_net_sales",
        "monthly_net_sales",
        "transactions",
        "customers",
        "qty",
        "avg_txn",
        "rolling_90",
        "rolling_180",
        "weekly_rolling_90",
        "weekly_rolling_180",
        "monthly_rolling_90",
        "monthly_rolling_180",
        "transactions_rolling_90",
        "transactions_rolling_180",
        "customers_rolling_90",
        "customers_rolling_180",
        "qty_rolling_90",
        "qty_rolling_180",
        "avg_txn_rolling_90",
        "avg_txn_rolling_180",
    ]]

    final["date"] = pd.to_datetime(final["date"]).dt.strftime("%Y-%m-%d")

    final.to_sql("high_level_daily", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
