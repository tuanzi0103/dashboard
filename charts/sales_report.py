import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from services.db import get_db
from datetime import datetime, timedelta, date  # 添加 date 导入
from services.category_rules import is_bar_category


def proper_round(x):
    """标准的四舍五入方法，0.5总是向上舍入"""
    if pd.isna(x):
        return x
    return round(x)


def persisting_multiselect(label, options, key, default=None):
    """持久化多选框，处理默认值不在选项中的情况"""
    if key not in st.session_state:
        st.session_state[key] = default or []

    # 过滤掉不在当前选项中的默认值
    st.session_state[key] = [item for item in st.session_state[key] if item in options]

    return st.multiselect(label, options, default=st.session_state[key], key=key)


def persisting_multiselect_with_width(label, options, key, default=None, width_chars=None):
    """持久化多选框，带宽度控制（与 high_level.py 一致）"""
    if key not in st.session_state:
        st.session_state[key] = default or []

    # 过滤掉不在当前选项中的默认值
    st.session_state[key] = [item for item in st.session_state[key] if item in options]

    # === 修改：添加自定义宽度参数 ===
    if width_chars is None:
        # 默认宽度为标签长度+1字符
        label_width = len(label)
        min_width = label_width + 1
    else:
        # 使用自定义宽度
        min_width = width_chars

    st.markdown(f"""
    <style>
        /* 强制设置多选框宽度 */
        [data-testid*="{key}"] {{
            width: {min_width}ch !important;
            min-width: {min_width}ch !important;
        }}
        [data-testid*="{key}"] > div {{
            width: {min_width}ch !important;
            min-width: {min_width}ch !important;
        }}
        [data-testid*="{key}"] [data-baseweb="select"] {{
            width: {min_width}ch !important;
            min-width: {min_width}ch !important;
        }}
        [data-testid*="{key}"] [data-baseweb="select"] > div {{
            width: {min_width}ch !important;
            min-width: {min_width}ch !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    return st.multiselect(label, options, default=st.session_state[key], key=key)


def _safe_sum(df, col):
    if df is None or df.empty or col not in df.columns:
        return 0.0
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return float(pd.to_numeric(s, errors="coerce").sum(skipna=True))
    s = (
        s.astype(str)
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .replace("", pd.NA)
    )
    return float(pd.to_numeric(s, errors="coerce").sum(skipna=True) or 0.0)


@st.cache_data(ttl=600, show_spinner=False)
def preload_all_data():
    """预加载所有需要的数据 - 与high_level.py相同的函数"""
    db = get_db()

    # 加载交易数据（包含日期信息）
    daily_sql = """
    WITH transaction_totals AS (
        SELECT 
            date(Datetime) AS date,
            [Transaction ID] AS txn_id,
            SUM([Gross Sales]) AS total_gross_sales,
            SUM(COALESCE(CAST(REPLACE(REPLACE([Tax], '$', ''), ',', '') AS REAL), 0)) AS total_tax,
            SUM(Qty) AS total_qty
        FROM transactions
        GROUP BY date, [Transaction ID]
    )
    SELECT
        date,
        SUM(ROUND(total_gross_sales - total_tax, 2)) AS net_sales_with_tax,
        SUM(total_gross_sales) AS gross_sales,
        SUM(total_tax) AS total_tax,
        COUNT(DISTINCT txn_id) AS transactions,
        CASE 
            WHEN COUNT(DISTINCT txn_id) > 0 
            THEN SUM(ROUND(total_gross_sales - total_tax, 2)) * 1.0 / COUNT(DISTINCT txn_id)
            ELSE 0 
        END AS avg_txn,
        SUM(total_qty) AS qty
    FROM transaction_totals
    GROUP BY date
    ORDER BY date;
    """

    category_sql = """
    WITH category_transactions AS (
        SELECT 
            date(Datetime) AS date,
            -- 修复：处理空分类，确保所有数据都被包含
            CASE 
                WHEN Category IS NULL OR TRIM(Category) = '' THEN 'None'
                ELSE TRIM(Category)
            END AS Category,
            [Transaction ID] AS txn_id,
            SUM([Net Sales]) AS cat_net_sales,
            SUM(COALESCE(CAST(REPLACE(REPLACE([Tax], '$', ''), ',', '') AS REAL), 0)) AS cat_tax,
            SUM([Gross Sales]) AS cat_gross,
            SUM(Qty) AS cat_qty
        FROM transactions
        GROUP BY date, Category, [Transaction ID]
    ),
    category_daily AS (
        SELECT
            date,
            Category,
            txn_id,
            SUM(ROUND(cat_net_sales + cat_tax, 2)) AS cat_total_with_tax,
            SUM(cat_net_sales) AS cat_net_sales,
            SUM(cat_tax) AS cat_tax,
            SUM(cat_gross) AS cat_gross,
            SUM(cat_qty) AS cat_qty
        FROM category_transactions
        GROUP BY date, Category, txn_id
    )
    SELECT
        date,
        Category,
        SUM(cat_total_with_tax) AS net_sales_with_tax,
        SUM(cat_net_sales) AS net_sales,
        SUM(cat_tax) AS total_tax,
        COUNT(DISTINCT txn_id) AS transactions,
        CASE 
            WHEN COUNT(DISTINCT txn_id) > 0 
            THEN SUM(cat_total_with_tax) * 1.0 / COUNT(DISTINCT txn_id)
            ELSE 0 
        END AS avg_txn,
        SUM(cat_gross) AS gross,
        SUM(cat_qty) AS qty
    FROM category_daily
    GROUP BY date, Category
    ORDER BY date, Category;
    """

    # 加载原始交易数据用于获取商品项（包含日期信息）
    item_sql = """
    SELECT 
        date(Datetime) as date,
        -- 修复：处理空分类
        CASE 
            WHEN Category IS NULL OR TRIM(Category) = '' THEN 'None'
            ELSE TRIM(Category)
        END AS Category,
        Item,
        [Net Sales],
        Tax,
        Qty,
        [Gross Sales]
    FROM transactions
    WHERE Item IS NOT NULL  -- 只排除空商品项，不排除空分类
    """

    daily = pd.read_sql(daily_sql, db)
    category = pd.read_sql(category_sql, db)
    items_df = pd.read_sql(item_sql, db)

    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")

        # 移除缺失数据的日期 (8.18, 8.19, 8.20) - 所有数据都过滤
        missing_dates = ['2025-08-18', '2025-08-19', '2025-08-20']
        daily = daily[~daily["date"].isin(pd.to_datetime(missing_dates))]

    if not category.empty:
        category["date"] = pd.to_datetime(category["date"])
        category = category.sort_values(["Category", "date"])

        # 移除缺失数据的日期 - 所有分类都过滤
        category = category[~category["date"].isin(pd.to_datetime(missing_dates))]

    if not items_df.empty:
        items_df["date"] = pd.to_datetime(items_df["date"])
        # 移除缺失数据的日期 - 商品数据也过滤
        items_df = items_df[~items_df["date"].isin(pd.to_datetime(missing_dates))]

    return daily, category, items_df


def extract_item_name(item):
    """提取商品名称，移除毫升/升等容量信息"""
    if pd.isna(item):
        return item

    # 移除容量信息（数字后跟ml/L等）
    import re
    # 匹配数字后跟ml/L/升/毫升等模式
    pattern = r'\s*\d+\.?\d*\s*(ml|mL|L|升|毫升)\s*$'
    cleaned = re.sub(pattern, '', str(item), flags=re.IGNORECASE)

    # 移除首尾空格
    return cleaned.strip()


def prepare_sales_data(df_filtered):

    # 复制数据避免修改原数据
    df = df_filtered.copy()

    df["final_sales"] = df.apply(
        lambda row: row["net_sales"] if is_bar_category(row["Category"]) else row["net_sales"],
        axis=1
    )
    return df


def extract_brand_name(item_name):
    """
    提取品牌：对清洗后的 Item 名称取第一个词作为品牌。
    这样像 "TLD Frenchs Forest Raw Honey 1Kg" -> "TLD"
          "HTG Organic Maple Syrup 1L" -> "HTG"
          "SPIRAL ORG Maple Syrup 250ml" -> "SPIRAL"
          "HANDHOE Macadamia Butter Roasted Crunchy 225g" -> "HANDHOE"
          "Beerose Honey 500g" -> "BEEROSE"
    避免把 'Butter/Honey/Maple/Jam/Tahini' 等产品词识别成品牌。
    """
    import re
    if pd.isna(item_name):
        return "Other"

    # 先用你已有的清洗函数做末尾规格/前缀清理
    cleaned = clean_item_name_for_comments(str(item_name))

    # 去掉多余空白
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "Other"

    # 按空格或连字符等分割
    tokens = re.split(r"[ \t\-_/]+", cleaned)

    # 取第一个"看起来像品牌"的 token：
    # - 至少含字母
    # - 非纯数字
    for tok in tokens:
        has_alpha = any(c.isalpha() for c in tok)
        if has_alpha and not tok.isdigit():
            # 清掉结尾的逗号/点号之类
            tok = tok.strip(",.;:()[]{}")
            if tok:
                return tok.upper()

    return "Other"


def calculate_item_sales(items_df, selected_categories, selected_items, start_date=None, end_date=None):
    """计算指定category和items的销售数据"""
    # 复制数据避免修改原数据
    filtered_items = items_df.copy()

    # 应用日期筛选
    if start_date is not None and end_date is not None:
        mask = (filtered_items["date"] >= pd.to_datetime(start_date)) & (
                filtered_items["date"] <= pd.Timestamp(end_date))
        filtered_items = filtered_items.loc[mask]

    # 如果有选中的分类，则应用分类筛选
    if selected_categories:
        filtered_items = filtered_items[filtered_items["Category"].isin(selected_categories)]

    # 清理商品名称用于匹配 - 移除所有计量单位
    filtered_items["clean_item"] = filtered_items["Item"].apply(clean_item_name_for_comments)

    # 如果有选中的商品，则应用商品项筛选
    if selected_items:
        filtered_items = filtered_items[filtered_items["clean_item"].isin(selected_items)]

    if filtered_items.empty:
        return pd.DataFrame()

    def calculate_sales(row):
        if is_bar_category(row["Category"]):
            return row["Net Sales"]
        else:
            return row["Net Sales"]

    filtered_items["final_sales"] = filtered_items.apply(calculate_sales, axis=1)

    # 按商品项汇总
    item_summary = filtered_items.groupby(["Category", "clean_item"]).agg({
        "Qty": "sum",
        "final_sales": "sum"
    }).reset_index()

    # === 修改：在汇总后进行四舍五入 ===
    item_summary["Qty"] = item_summary["Qty"].apply(lambda x: int(proper_round(x)) if pd.notna(x) else 0)
    item_summary["final_sales"] = item_summary["final_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

    return item_summary.rename(columns={
        "clean_item": "Item",
        "Qty": "Sum of Items Sold",
        "final_sales": "Sum of Daily Sales"
    })[["Category", "Item", "Sum of Items Sold", "Sum of Daily Sales"]]


def calculate_item_daily_trends(items_df, selected_categories, selected_items, start_date=None, end_date=None):
    """计算指定category和items的每日趋势数据"""
    # 复制数据避免修改原数据
    filtered_items = items_df.copy()

    # 应用日期筛选
    if start_date is not None and end_date is not None:
        mask = (filtered_items["date"] >= pd.to_datetime(start_date)) & (
                filtered_items["date"] <= pd.Timestamp(end_date))
        filtered_items = filtered_items.loc[mask]

    # 如果有选中的分类，则应用分类筛选
    if selected_categories:
        filtered_items = filtered_items[filtered_items["Category"].isin(selected_categories)]

    # 清理商品名称用于匹配 - 移除所有计量单位
    filtered_items["clean_item"] = filtered_items["Item"].apply(clean_item_name_for_comments)

    # 如果有选中的商品，则应用商品项筛选
    if selected_items:
        filtered_items = filtered_items[filtered_items["clean_item"].isin(selected_items)]

    if filtered_items.empty:
        return pd.DataFrame()

    # 计算每个商品项的销售数据
    def calculate_sales(row):
        # === 修改：所有Bar分类也使用Net Sales（不含税）===
        if is_bar_category(row["Category"]):
            # Bar分类：现在只使用Net Sales（不含税）
            return row["Net Sales"]  # 不再在这里四舍五入
        else:
            # 非Bar分类：直接使用Net Sales
            return row["Net Sales"]  # 不再在这里四舍五入

    filtered_items["final_sales"] = filtered_items.apply(calculate_sales, axis=1)

    # 按日期和商品项汇总
    daily_trends = filtered_items.groupby(["date", "Category", "clean_item"]).agg({
        "Qty": "sum",
        "final_sales": "sum"
    }).reset_index()

    # 按日期汇总所有选中商品的总和
    daily_summary = daily_trends.groupby("date").agg({
        "Qty": "sum",
        "final_sales": "sum"
    }).reset_index()

    # === 修改：在汇总后进行四舍五入 ===
    daily_summary["Qty"] = daily_summary["Qty"].apply(lambda x: int(proper_round(x)) if pd.notna(x) else 0)
    daily_summary["final_sales"] = daily_summary["final_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

    return daily_summary.rename(columns={
        "Qty": "Sum of Items Sold",
        "final_sales": "Sum of Daily Sales"
    })[["date", "Sum of Items Sold", "Sum of Daily Sales"]]


def clean_item_name_for_comments(item):
    """清理商品名称 - 移除所有计量单位但保留商品名"""
    if pd.isna(item):
        return item

    # 移除所有类型的计量单位（重量、容量等）
    import re
    # 匹配数字后跟g/kg/ml/L/升/毫升/oz/lb等模式，移除整个计量单位部分
    pattern = r'\s*\d+\.?\d*\s*(g|kg|ml|mL|L|升|毫升|oz|lb)\s*$'
    cleaned = re.sub(pattern, '', str(item), flags=re.IGNORECASE)

    # 移除所有 "XXX - " 这种前缀模式（比如 "$460 WRAP -", "$360 BREAKFAST -", "$345 BURRITO -"）
    cleaned = re.sub(r'^.*?[a-zA-Z]+\s*-\s*', '', cleaned)

    # 移除首尾空格
    cleaned = cleaned.strip()

    return cleaned


def get_top_items_by_category(items_df, categories, start_date=None, end_date=None, for_total=False):
    """获取每个分类销量前3的商品，按品牌分组
    for_total: 如果为True，则返回整个分类组的前3品牌
    """
    if not categories:
        return {}

    # 复制数据避免修改原数据
    filtered_items = items_df.copy()

    # 应用日期筛选
    if start_date is not None and end_date is not None:
        mask = (filtered_items["date"] >= pd.to_datetime(start_date)) & (
                filtered_items["date"] <= pd.Timestamp(end_date))
        filtered_items = filtered_items.loc[mask]

    # 过滤指定分类的商品
    filtered_items = filtered_items[filtered_items["Category"].isin(categories)]

    if filtered_items.empty:
        return {}

    # 计算每个商品项的销售数据
    def calculate_sales(row):
        # === 修改：所有Bar分类也使用Net Sales（不含税）===
        if is_bar_category(row["Category"]):
            # Bar分类：现在只使用Net Sales（不含税）
            return row["Net Sales"]  # 不再在这里四舍五入
        else:
            # 非Bar分类：直接使用Net Sales
            return row["Net Sales"]  # 不再在这里四舍五入

    filtered_items["final_sales"] = filtered_items.apply(calculate_sales, axis=1)

    # 清理商品名称 - 移除所有计量单位
    filtered_items["clean_item"] = filtered_items["Item"].apply(clean_item_name_for_comments)

    # 提取品牌名称 - 使用改进的品牌检测
    filtered_items["brand"] = filtered_items["clean_item"].apply(extract_brand_name)

    if for_total:
        # 对于总计行，获取整个分类组的前3品牌
        brand_sales = filtered_items.groupby("brand").agg({
            "final_sales": "sum"
        }).reset_index()

        # === 修改：在汇总后进行四舍五入 ===
        brand_sales["final_sales"] = brand_sales["final_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

        if not brand_sales.empty:
            top_3 = brand_sales.nlargest(3, "final_sales")
            # 格式：$销售额 品牌名
            top_brands_list = [f"${int(row['final_sales'])} {row['brand']}" for _, row in top_3.iterrows()]
            return ", ".join(top_brands_list)
        else:
            return "No items"
    else:
        # 对于普通行，获取每个分类的前3品牌
        category_brands = filtered_items.groupby(["Category", "brand"]).agg({
            "final_sales": "sum"
        }).reset_index()

        # === 修改：在汇总后进行四舍五入 ===
        category_brands["final_sales"] = category_brands["final_sales"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

        # 获取每个分类的前3品牌
        top_brands_by_category = {}
        for category in categories:
            category_data = category_brands[category_brands["Category"] == category]
            if not category_data.empty:
                top_3 = category_data.nlargest(3, "final_sales")
                # 格式：$销售额 品牌名
                top_brands_list = [f"${int(row['final_sales'])} {row['brand']}" for _, row in top_3.iterrows()]
                top_brands_by_category[category] = ", ".join(top_brands_list)
            else:
                top_brands_by_category[category] = "No items"

        return top_brands_by_category


def show_sales_report(tx: pd.DataFrame, inv: pd.DataFrame):
    # === 全局样式: 让 st.dataframe 里的所有表格文字左对齐 ===
    st.markdown("""
    <style>
    [data-testid="stDataFrame"] table {
        text-align: left !important;
    }
    [data-testid="stDataFrame"] th {
        text-align: left !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stDataFrame"] td {
        text-align: left !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style='font-size:22px; font-weight:700; margin-top:-2rem !important; margin-bottom:0.2rem !important;'>🧾 Sales Report by Category</h2>
    <style>
    /* 去掉 Streamlit 默认标题和上一个元素之间的间距 */
    div.block-container h2 {
        padding-top: 0 !important;
        margin-top: -2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 预加载所有数据 - 使用与high_level.py相同的数据源
    with st.spinner("Loading data..."):
        daily, category_tx, items_df = preload_all_data()

    # 在这里添加初始化代码
    if "bar_items_select" not in st.session_state:
        st.session_state["bar_items_select"] = []
    if "retail_items_select" not in st.session_state:
        st.session_state["retail_items_select"] = []

    if category_tx.empty:
        st.info("No category data available.")
        return

    # ---------------- Time Range Filter ----------------
    st.markdown("<h4 style='font-size:16px; font-weight:700;'>📅 Time Range</h4>", unsafe_allow_html=True)

    # 🔹 使用三列布局缩短下拉框宽度，与 high_level.py 保持一致
    col1, col2, col3, _ = st.columns([1, 1, 1, 4])

    with col1:
        # 应用与 high_level.py 相同的选择框样式
        range_opt = st.selectbox("Select range", ["Custom dates", "WTD", "MTD", "YTD"], key="sr_range")

    today = pd.Timestamp.today().normalize()
    start_date, end_date = None, today

    if range_opt == "Custom dates":
        # ==== ✅ 自动计算最近有数据的一周 ====
        if not category_tx.empty:
            all_dates = sorted(category_tx["date"].dt.normalize().unique())
            today = pd.Timestamp.today().normalize()
            this_monday = today - pd.Timedelta(days=today.weekday())  # 当前周一
            this_sunday = this_monday + pd.Timedelta(days=6)

            # 当前周的日期范围
            this_week_mask = (category_tx["date"] >= this_monday) & (category_tx["date"] <= this_sunday)
            this_week_data = category_tx.loc[this_week_mask]

            if not this_week_data.empty:
                # ✅ 当前周有数据，默认显示当前周
                default_from, default_to = this_monday, min(this_sunday, all_dates[-1])
            else:
                # ✅ 当前周无数据，则回退上一周
                last_monday = this_monday - pd.Timedelta(days=7)
                last_sunday = this_sunday - pd.Timedelta(days=7)
                last_week_mask = (category_tx["date"] >= last_monday) & (category_tx["date"] <= last_sunday)
                last_week_data = category_tx.loc[last_week_mask]
                if not last_week_data.empty:
                    default_from, default_to = last_monday, last_sunday
                else:
                    # 如果两周都没数据，则取最近有数据的一周
                    latest_date = pd.to_datetime(all_dates[-1])
                    default_to = latest_date
                    default_from = latest_date - pd.Timedelta(days=6)
        else:
            # 数据为空时回退默认
            today = pd.Timestamp.today().normalize()
            default_from, default_to = today - pd.Timedelta(days=7), today

        # ========== 在这里添加类型转换代码 ==========
        # ========== 在这里添加类型转换代码 ==========
        # 确保默认日期是 date 类型
        def ensure_date_type(date_obj):
            """确保日期对象是 Python date 类型"""
            if date_obj is None:
                return None
            if isinstance(date_obj, pd.Timestamp):
                return date_obj.date()
            if isinstance(date_obj, datetime):
                return date_obj.date()
            if isinstance(date_obj, date):
                return date_obj
            # 处理 numpy.datetime64 类型
            if isinstance(date_obj, np.datetime64):
                return pd.Timestamp(date_obj).date()
            # 如果是字符串，尝试转换
            if isinstance(date_obj, str):
                try:
                    return pd.to_datetime(date_obj).date()
                except:
                    return date_obj
            return date_obj

        # === 日期选择器 ===
        col_from, col_to, _ = st.columns([1, 1, 5])

        # 🔧 Step 1：先修复默认值顺序（确保输入框显示正常）
        if default_from > default_to:
            default_from, default_to = default_to, default_from

        with col_from:
            # 确保是 Python date 类型
            if not isinstance(default_from, date):
                try:
                    default_from = pd.Timestamp(default_from).date()
                except:
                    default_from = date.today() - timedelta(days=7)

            t1 = st.date_input(
                "From",
                value=default_from,
                key="sr_date_from",
                format="DD/MM/YYYY"
            )
        with col_to:
            # 确保是 Python date 类型
            if not isinstance(default_to, date):
                try:
                    default_to = pd.Timestamp(default_to).date()
                except:
                    default_to = date.today()

            t2 = st.date_input(
                "To",
                value=default_to,
                key="sr_date_to",
                format="DD/MM/YYYY"
            )

        # 🔧 修复：如果用户选择或默认计算导致 t1 > t2，自动交换
        if t1 > t2:
            t1, t2 = t2, t1

        if t1 and t2:
            start_date, end_date = pd.to_datetime(t1), pd.to_datetime(t2)

    elif range_opt == "WTD":
        start_date = today - pd.Timedelta(days=today.weekday())
    elif range_opt == "MTD":
        start_date = today.replace(day=1)
    elif range_opt == "YTD":
        start_date = today.replace(month=1, day=1)

    # 应用时间范围筛选到category数据
    df_filtered = category_tx.copy()
    if start_date is not None and end_date is not None:
        mask = (df_filtered["date"] >= pd.to_datetime(start_date)) & (
                df_filtered["date"] <= pd.Timestamp(end_date))
        df_filtered = df_filtered.loc[mask]

    # 应用数据修复
    df_filtered_fixed = prepare_sales_data(df_filtered)

    # ---------------- Bar Charts ----------------
    # 使用修复后的数据
    g = df_filtered_fixed.groupby("Category", as_index=False).agg(
        items_sold=("qty", "sum"),
        daily_sales=("final_sales", "sum")  # 使用修复后的销售额
    ).sort_values("items_sold", ascending=False)

    g = g[g["Category"] != "None"]

    if not g.empty:
        c1, c2 = st.columns(2)
        with c1:
            # 只显示Top 10分类
            g_top10_items = g.head(10)
            fig1 = px.bar(g_top10_items, x="Category", y="items_sold", title="Items Sold (by Category) - Top 10",
                          height=400)
            fig1.update_layout(margin=dict(t=60, b=60))
            st.plotly_chart(fig1, config={"responsive": True, "displayModeBar": True})

        with c2:
            # 只显示Top 10分类
            g_sorted = g.sort_values("daily_sales", ascending=False).head(10)
            fig2 = px.bar(g_sorted, x="Category", y="daily_sales", title="Daily Sales (by Category) - Top 10",
                          height=400)
            fig2.update_layout(margin=dict(t=60, b=60))
            st.plotly_chart(fig2, config={"responsive": True, "displayModeBar": True})
    else:
        st.info("No data under current filters.")
        return

    all_cats = df_filtered_fixed["Category"].dropna().astype(str).unique().tolist()

    bar_cats = [c for c in all_cats if is_bar_category(c)]
    retail_cats = [c for c in all_cats if not is_bar_category(c)]

    def time_range_summary(data, cats, range_type, start_dt, end_dt):
        # 确保包含所有指定的分类，即使当天没有销售数据
        # 先创建一个包含所有分类的空DataFrame作为基础
        all_cats_df = pd.DataFrame({"Category": list(cats)})

        # 获取当天的数据
        sub = data[data["Category"].isin(cats)].copy()

        # 合并所有分类，确保即使没有销售数据的分类也包含在内
        summary = all_cats_df.merge(sub.groupby("Category", as_index=False).agg(
            items_sold=("qty", "sum"),
            daily_sales=("final_sales", "sum")
        ), on="Category", how="left")

        # 填充缺失值
        summary["items_sold"] = summary["items_sold"].fillna(0)
        summary["daily_sales"] = summary["daily_sales"].fillna(0)

        # 计算与前一个相同长度时间段的对比
        if start_dt and end_dt:
            # === 新增逻辑：如果选择的是同一天，则与前一天比较 ===
            is_single_day = (start_dt.date() == end_dt.date())

            if is_single_day:
                # ✅ 单日逻辑：使用前一天的数据进行比较
                prev_day = start_dt - timedelta(days=1)
                prev_start = prev_day
                prev_end = prev_day
            else:
                # ✅ 正常时间段逻辑：与前一个相同长度时间段比较
                time_diff = end_dt - start_dt
                prev_start = start_dt - time_diff - timedelta(days=1)
                prev_end = start_dt - timedelta(days=1)

            # 获取前一个时间段的数据 - 直接从原始数据获取，确保数据完整
            prev_mask = (category_tx["date"] >= pd.to_datetime(prev_start)) & (
                        category_tx["date"] <= pd.to_datetime(prev_end))
            prev_data_raw = category_tx.loc[prev_mask].copy()

            # 对历史数据也应用相同的修复逻辑
            prev_data_fixed = prepare_sales_data(prev_data_raw)

            if not prev_data_fixed.empty:
                # 确保只获取指定分类的数据
                prev_data_filtered = prev_data_fixed[prev_data_fixed["Category"].isin(cats)]
                prev_summary = prev_data_filtered.groupby("Category", as_index=False).agg(
                    prior_daily_sales=("final_sales", "sum")  # 使用修复后的销售额
                )

                # 合并前一天数据，确保所有分类都包含
                summary = summary.merge(prev_summary, on="Category", how="left")
                summary["prior_daily_sales"] = summary["prior_daily_sales"].fillna(0)

                # 调试总销售额
                total_prior = summary["prior_daily_sales"].sum()
            else:
                summary["prior_daily_sales"] = 0
        else:
            summary["prior_daily_sales"] = 0

        # === 修改：保留原始 daily_sales 精度，用于 Total 汇总 ===
        summary["daily_sales_raw"] = summary["daily_sales"]  # 保存原始浮点值供后续计算
        MIN_BASE = 50

        # === 修正 weekly change ===
        # 检测是否选择了单日
        # === 修正 weekly change ===
        # 检测是否选择了单日
        is_single_day = (start_dt is not None and end_dt is not None and start_dt.date() == end_dt.date())

        if is_single_day:
            # ✅ 单日逻辑：使用前一天的数据进行比较 (10.29 vs 10.28)
            summary["weekly_change"] = np.where(
                summary["prior_daily_sales"] > MIN_BASE,
                (summary["daily_sales_raw"] - summary["prior_daily_sales"]) / summary["prior_daily_sales"] * 100,
                np.nan
            )

        else:
            # ✅ 正常时间段逻辑：与前一个相同长度时间段比较
            summary["weekly_change"] = np.where(
                summary["prior_daily_sales"] > MIN_BASE,
                (summary["daily_sales"] - summary["prior_daily_sales"]) / summary["prior_daily_sales"] * 100,
                np.nan
            )

        # 计算日均销量
        if start_dt and end_dt:
            days_count = (end_dt - start_dt).days + 1
            summary["per_day"] = summary["items_sold"] / days_count
        else:
            summary["per_day"] = summary["items_sold"] / 7  # 默认按7天计算

        # 仅 items_sold 取整
        summary["items_sold"] = summary["items_sold"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

        # 展示列用整数，但不影响 raw 精度
        summary["daily_sales_display"] = summary["daily_sales"].apply(
            lambda x: proper_round(x) if pd.notna(x) else x
        ).astype(int)

        # per_day 也取整展示
        summary["per_day"] = summary["per_day"].apply(lambda x: proper_round(x) if pd.notna(x) else x)

        return summary

    # helper: 格式化 + 高亮
    def format_change(x):
        if pd.isna(x):
            return "N/A"
        return f"{x * 100:+.2f}%"

    def highlight_change(val):
        if val == "N/A":
            color = "gray"
        elif val.startswith("+"):
            color = "green"
        elif val.startswith("-"):
            color = "red"
        else:
            color = "black"
        return f"color: {color}"

    # ---------------- Bar table ----------------
    st.markdown("<h4 style='font-size:16px; font-weight:700;'>📊 Bar Categories</h4>", unsafe_allow_html=True)


    bar_df = time_range_summary(df_filtered_fixed, bar_cats, range_opt, start_date, end_date)
    if not bar_df.empty:
        # 获取Bar分类的前3品牌
        bar_top_items = get_top_items_by_category(items_df, bar_cats, start_date, end_date, for_total=False)
        # 获取Bar分类组的前3品牌（用于总计行）
        bar_total_top_items = get_top_items_by_category(items_df, bar_cats, start_date, end_date, for_total=True)

        # 添加Comments列
        bar_df["Comments"] = bar_df["Category"].map(bar_top_items)

        # ✅ 用整数显示，避免小数+红角
        bar_df["daily_sales_display"] = bar_df["daily_sales"].apply(
            lambda x: proper_round(x) if pd.notna(x) else x).astype(int)

        # ✅ raw 数值列用于排序、避免红角
        bar_df["daily_sales_raw"] = bar_df["daily_sales"]

        bar_df = bar_df.rename(columns={
            "Category": "Row Labels",
            "items_sold": "Sum of Items Sold",
            "daily_sales_display": "Sum of Daily Sales",  # ✅ 用展示列
            "weekly_change": "Weekly change",
            "per_day": "Per day"
        })

        # ✅ format & sort columns for Sum of Daily Sales
        bar_df["_sort_daily_sales"] = bar_df["Sum of Daily Sales"]
        bar_df["Sum of Daily Sales Display"] = bar_df["Sum of Daily Sales"].apply(lambda x: f"${int(x)}")

        bar_df = bar_df.sort_values("Sum of Daily Sales", ascending=False)
        # 创建总计行
        total_items_sold = bar_df["Sum of Items Sold"].sum()
        # === 修复：使用原始精度计算，不要提前四舍五入 ===
        total_daily_sales_raw = bar_df["daily_sales_raw"].sum()  # 使用原始浮点值
        total_per_day = bar_df["Per day"].sum()

        # 计算Total行的Weekly change - 基于总销售额与前一周期的对比
        total_prior_sales = bar_df["prior_daily_sales"].sum()
        MIN_BASE = 50
        if total_prior_sales > MIN_BASE:
            total_weekly_change = (total_daily_sales_raw - total_prior_sales) / total_prior_sales * 100
        else:
            total_weekly_change = np.nan

        # 显示时再四舍五入
        total_daily_sales = proper_round(total_daily_sales_raw)
        total_daily_sales_display = f"${total_daily_sales:,.0f}"

        # === 创建数据框（与high_level.py相同的格式）- 总计行放在第一行 ===
        bar_summary_data = {
            'Row Labels': ["Total"] + bar_df["Row Labels"].tolist(),
            'Sum of Items Sold': [total_items_sold] + bar_df["Sum of Items Sold"].tolist(),
            'Sum of Daily Sales': [total_daily_sales_display] + bar_df["Sum of Daily Sales Display"].tolist(),
            '_sort_daily_sales': [total_daily_sales] + bar_df["_sort_daily_sales"].tolist(),

            'Weekly change': [total_weekly_change] + bar_df["Weekly change"].tolist(),
            'Per day': [total_per_day] + bar_df["Per day"].tolist(),
            'Comments': [bar_total_top_items] + bar_df["Comments"].tolist()
        }

        df_bar_summary = pd.DataFrame(bar_summary_data)

        # === 修正：直接按照Weekly change数值从小到大排序 ===
        # 先分离Total行和其他行
        total_row = df_bar_summary[df_bar_summary['Row Labels'] == 'Total']
        other_rows = df_bar_summary[df_bar_summary['Row Labels'] != 'Total']
        # 直接按 Weekly change 排序
        other_rows_sorted = other_rows.sort_values(
            by='Weekly change',
            key=lambda x: pd.to_numeric(x, errors='coerce'),
            ascending=True,
            na_position='last'
        )

        df_bar_summary_sorted = pd.concat([total_row, other_rows_sorted], ignore_index=True)

        # === ✅ 保持等宽且保留自定义列宽 ===
        TABLE_WIDTH = 730

        bar_column_config = {
            "Row Labels": st.column_config.Column(width=130),
            "Sum of Items Sold": st.column_config.NumberColumn("Sum of Items Sold", width=110, format="%d"),
            "Sum of Daily Sales": st.column_config.NumberColumn(  # 改为 NumberColumn
                "Sum of Daily Net Sales",
                width=130,
                format="%d"  # 去掉千位分隔符，直接显示数字
            ),
            "_sort_daily_sales": st.column_config.NumberColumn("", width=1, format="%d"),
            "Per day": st.column_config.NumberColumn("Per day", width=70, format="%d"),
            "Comments": st.column_config.Column(width=240),
            "Weekly change": st.column_config.NumberColumn(width=100, label="Weekly change", format="%.2f%%"),
        }

        # === 固定宽度的CSS，不改列宽比例，只统一外框 ===
        st.markdown(f"""
        <style>
        .bar-table-wrapper {{
            width:{TABLE_WIDTH}px !important;
            max-width:{TABLE_WIDTH}px !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .bar-table-wrapper [data-testid="stDataFrame"] {{
            width:{TABLE_WIDTH}px !important;
            max-width:{TABLE_WIDTH}px !important;
            min-width:{TABLE_WIDTH}px !important;
            overflow-x:hidden !important;
        }}
        .bar-table-wrapper [data-testid="stDataFrame"] table {{
            table-layout: fixed !important;
            width:{TABLE_WIDTH}px !important;
        }}
        .bar-table-wrapper [data-testid="stDataFrame"] td,
        .bar-table-wrapper [data-testid="stDataFrame"] th {{
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        # === 两个表放在同一个容器 ===
        st.markdown(f"<div class='bar-table-wrapper' style='border: 0.1px solid #e6e6e6; padding: 0px; margin: 0px;'>",
                    unsafe_allow_html=True)

        st.dataframe(
            total_row[["Row Labels", "Sum of Items Sold", "Sum of Daily Sales",
                       "Per day", "Comments", "Weekly change"]],
            column_config=bar_column_config,
            hide_index=True,
            width=800
        )
        # === 添加：减少两个表格之间的间距 ===
        st.markdown("""
        <style>
        [data-testid="stDataFrame"] {
            margin-top: -16px !important;
            margin-bottom: -16px !important;
        }
        </style>
        """, unsafe_allow_html=True)


        # 主表 - 隐藏排序列
        st.dataframe(
            other_rows_sorted[["Row Labels", "Sum of Items Sold", "Sum of Daily Sales",
                               "Per day", "Comments", "Weekly change"]],
            column_config=bar_column_config,
            hide_index=True,
            width=800
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Bar分类商品项选择 - 使用与 high_level.py 相同的多选框样式
        st.markdown("<h4 style='font-size:16px; font-weight:700;'>📦 Bar Category Items</h4>", unsafe_allow_html=True)

        # 获取所有Bar分类的商品项
        bar_items_df = items_df[
            items_df["Category"].apply(is_bar_category)
        ].copy()

        if not bar_items_df.empty:
            # 使用新的清理函数移除所有计量单位
            bar_items_df["clean_item"] = bar_items_df["Item"].apply(clean_item_name_for_comments)
            bar_item_options = sorted(bar_items_df["clean_item"].dropna().unique())

            # 选择Bar分类和商品项 - 放在同一行
            col_bar1, col_bar2, col_bar3, _ = st.columns([1.2, 1.6, 1.3, 2.9])
            with col_bar1:
                selected_bar_categories = persisting_multiselect_with_width(
                    "Select Bar Categories",
                    options=sorted(bar_df["Row Labels"].unique()),
                    key="bar_categories_select",
                    width_chars=22
                )
            with col_bar2:
                # 为商品项选择创建表单，避免立即 rerun
                with st.form(key="bar_items_form"):
                    selected_bar_items = st.multiselect(
                        "Select Items from Bar Categories",
                        options=bar_item_options,
                        default=st.session_state.get("bar_items_select", []),
                        key="bar_items_widget"
                    )

                    # 应用按钮
                    submitted_bar = st.form_submit_button("Apply", type="primary")

                    if submitted_bar:
                        # 更新 session state
                        st.session_state["bar_items_select"] = selected_bar_items
                        st.rerun()

                # 从 session state 获取最终的选择
                selected_bar_items = st.session_state.get("bar_items_select", [])

                # 显示当前选择状态
                if selected_bar_items:
                    st.caption(f"✅ Selected: {len(selected_bar_items)} items")
                else:
                    st.caption("ℹ️ No items selected")

            # 显示选中的商品项数据
            if selected_bar_categories or selected_bar_items:
                bar_item_summary = calculate_item_sales(
                    items_df, selected_bar_categories, selected_bar_items, start_date, end_date
                )



                if not bar_item_summary.empty:
                    # 设置列配置
                    item_column_config = {
                        'Category': st.column_config.Column(width="150px"),
                        'Item': st.column_config.Column(width="160px"),
                        'Sum of Items Sold': st.column_config.Column(width="120px"),
                        'Sum of Daily Sales': st.column_config.Column(width="90px")
                    }

                    st.data_editor(
                        bar_item_summary,
                        column_config=item_column_config,
                        use_container_width=False,
                        disabled=True,
                        hide_index=True
                    )

                    # 显示小计
                    total_qty = bar_item_summary["Sum of Items Sold"].sum()
                    total_sales = bar_item_summary["Sum of Daily Sales"].sum()
                    st.write(f"**Subtotal for selected items:** {total_qty} items, ${total_sales:,.0f}")


                    # 显示每日趋势柱形图（并列样式 + 图表宽度缩小为原来的一半）
                    bar_daily_trends = calculate_item_daily_trends(
                        items_df, selected_bar_categories, selected_bar_items, start_date, end_date
                    )
                    bar_daily_trends["date_str"] = bar_daily_trends["date"].dt.strftime("%Y-%m-%d")

                    if not bar_daily_trends.empty:
                        # ✅ 多选框宽度，与 Select Bar Categories 一致
                        metric_col1, _ = st.columns([1.5, 5.5])
                        with metric_col1:
                            metric_option = persisting_multiselect_with_width(
                                label="Select metrics to display:",
                                options=["Sum of Items Sold", "Sum of Daily Sales"],
                                key="bar_daily_metric_select",
                                default=["Sum of Items Sold", "Sum of Daily Sales"],
                                width_chars=25  # 🔧 控制多选框宽度
                            )

                        # === 创建双轴 & 分组柱图 ===
                        fig = go.Figure()

                        # --- 左轴：Items Sold ---
                        if "Sum of Items Sold" in metric_option:
                            fig.add_trace(go.Bar(
                                x=bar_daily_trends["date_str"],
                                y=bar_daily_trends["Sum of Items Sold"],
                                name="Sum of Items Sold",
                                marker_color="#4F6D7A",
                                #width=0.35,
                                yaxis="y",  # ★ 左轴
                                offsetgroup=0  # ★ 分组编号 0
                            ))

                        # --- 右轴：Daily Sales ---
                        if "Sum of Daily Sales" in metric_option:
                            fig.add_trace(go.Bar(
                                x=bar_daily_trends["date_str"],
                                y=bar_daily_trends["Sum of Daily Sales"],
                                name="Sum of Daily Sales ($)",
                                marker_color="#F2A65A",
                                #width=0.35,
                                yaxis="y2",  # ★ 右轴
                                offsetgroup=1  # ★ 分组编号 1 → 不会重叠！
                            ))

                        show_items = "Sum of Items Sold" in metric_option
                        show_sales = "Sum of Daily Sales" in metric_option

                        # 网格逻辑：
                        # - 两个都选 → 网格来自左轴
                        # - 只选 Items Sold → 左轴网格
                        # - 只选 Daily Sales → 右轴网格
                        left_grid = show_items  # 左轴画网格？
                        right_grid = (show_sales and not show_items)  # 右轴画网格？

                        # === 布局：双纵轴 + 分组柱 ===
                        fig.update_layout(
                            title="Daily Trends for Selected Items",
                            xaxis=dict(title="Date"),

                            # 左轴
                            yaxis=dict(
                                title="Items Sold",
                                showgrid=left_grid,
                                zeroline=True
                            ),

                            # 右轴
                            yaxis2=dict(
                                title="Sales ($)",
                                overlaying="y",
                                side="right",
                                showgrid=right_grid  # ★ 当只有 sales 时自动开启
                            ),

                            barmode="group",  # ★ 分组显示（两个柱并排）
                            bargap=0.15,
                            bargroupgap=0.1,
                            height=420,
                            margin=dict(t=60, b=60),
                            hovermode="x unified",

                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        # ★ 修复长日期范围截断 —— 强制 datetime 轴格式
                        fig.update_xaxes(tickformat="%b %d", tickangle=-45)

                        # ✅ 仅当只显示 Items Sold（未勾选 Sales）时，强制整数并自适配步长
                        if ("Sum of Items Sold" in metric_option) and ("Sum of Daily Sales" not in metric_option):
                            qty_max = int(max(1, (
                                bar_daily_trends if 'bar_daily_trends' in locals() else bar_daily_trends)[
                                "Sum of Items Sold"].max()))

                            # 目标 5~7 个刻度，选择 1/2/5×10^k 的“漂亮步长”
                            import math
                            def nice_dtick(n_max, target_ticks=6):
                                raw = max(1, math.ceil(n_max / target_ticks))
                                bases = [1, 2, 5]
                                k = 0
                                while True:
                                    for b in bases:
                                        step = b * (10 ** k)
                                        if step >= raw:
                                            return step
                                    k += 1

                            step = nice_dtick(qty_max)
                            fig.update_yaxes(
                                tickmode="linear",
                                tick0=0,
                                dtick=step,  # ← 关键：1/2/5×10^k 自适应
                                rangemode="tozero"  # 从 0 起，最大值交给 Plotly 自动算
                            )

                        # ✅ 图表居中显示，宽度为页面一半
                        chart_col1, _ = st.columns([1, 1])
                        with chart_col1:
                            st.plotly_chart(fig, config={"responsive": True, "displayModeBar": True})
                else:
                    st.info("No data for selected items.")
        else:
            st.info("No items found in Bar categories.")

    else:
        st.info("No data for Bar categories.")

    # ---------------- Retail table ----------------
    st.markdown("<h4 style='font-size:16px; font-weight:700;'>🛍️ Retail Categories</h4>", unsafe_allow_html=True)
    retail_df = time_range_summary(df_filtered_fixed, retail_cats, range_opt, start_date, end_date)

    if not retail_df.empty:
        # 获取Retail分类的前3品牌
        retail_top_items = get_top_items_by_category(items_df, retail_cats, start_date, end_date, for_total=False)
        # 获取Retail分类组的前3品牌（用于总计行）
        retail_total_top_items = get_top_items_by_category(items_df, retail_cats, start_date, end_date, for_total=True)

        # 添加Comments列
        retail_df["Comments"] = retail_df["Category"].map(retail_top_items)

        retail_df = retail_df.rename(columns={
            "Category": "Row Labels",
            "items_sold": "Sum of Items Sold",
            "daily_sales_display": "Sum of Daily Sales",  # ✅ 改为用取整展示列
            "weekly_change": "Weekly change",
            "per_day": "Per day"
        })

        retail_df = retail_df.sort_values("Sum of Daily Sales", ascending=False)

        # 创建总计行
        # === 修复：先用原始浮点数计算百分比，再四舍五入显示 ===
        total_daily_sales_raw = retail_df["daily_sales_raw"].sum()
        total_prior_sales_raw = retail_df["prior_daily_sales"].sum()
        MIN_BASE = 50
        if total_prior_sales_raw > MIN_BASE:
            total_weekly_change = (total_daily_sales_raw - total_prior_sales_raw) / total_prior_sales_raw * 100  # 乘以100
        else:
            total_weekly_change = np.nan

        # 显示时再四舍五入
        total_items_sold = proper_round(retail_df["Sum of Items Sold"].sum())
        total_daily_sales = proper_round(total_daily_sales_raw)
        total_per_day = proper_round(retail_df["Per day"].sum())

        # === 修复：创建带千位分隔符的显示列和隐藏的排序列 ===
        retail_df["Sum of Daily Sales Display"] = retail_df["Sum of Daily Sales"].apply(lambda x: f"${int(x)}")
        retail_df["_sort_daily_sales"] = retail_df["Sum of Daily Sales"]  # 隐藏的数值列用于排序
        total_daily_sales_display = f"${int(total_daily_sales)}"

        # 创建数据框（与high_level.py相同的格式）- 总计行放在第一行
        retail_summary_data = {
            'Row Labels': ["Total"] + retail_df["Row Labels"].tolist(),
            'Sum of Items Sold': [total_items_sold] + retail_df["Sum of Items Sold"].tolist(),
            'Sum of Daily Sales': [total_daily_sales_display] + retail_df["Sum of Daily Sales Display"].tolist(),
            # 使用带千位分隔符的显示列
            '_sort_daily_sales': [total_daily_sales] + retail_df["_sort_daily_sales"].tolist(),  # 隐藏的数值列用于排序
            'Weekly change': [total_weekly_change] + retail_df["Weekly change"].tolist(),
            'Per day': [total_per_day] + retail_df["Per day"].tolist(),
            'Comments': [retail_total_top_items] + retail_df["Comments"].tolist()
        }

        df_retail_summary = pd.DataFrame(retail_summary_data)

        # === 修正：直接按照Weekly change数值从小到大排序 ===
        total_row = df_retail_summary[df_retail_summary['Row Labels'] == 'Total']
        other_rows = df_retail_summary[df_retail_summary['Row Labels'] != 'Total']

        # 直接按 Weekly change 排序
        other_rows_sorted = other_rows.sort_values(
            by='Weekly change',
            key=lambda x: pd.to_numeric(x, errors='coerce'),
            ascending=True,
            na_position='last'
        )

        # Total 行始终放在最上方
        df_retail_summary_sorted = pd.concat([total_row, other_rows_sorted], ignore_index=True)

        # === ✅ Retail Category: Total单独列出 + 灰线 + 保持列宽一致 ===
        TABLE_WIDTH = 730  # 跟Bar保持一致

        # === 拆分 Total 与其他行 ===
        total_row_retail = df_retail_summary_sorted[df_retail_summary_sorted['Row Labels'] == 'Total']
        other_rows_retail = df_retail_summary_sorted[df_retail_summary_sorted['Row Labels'] != 'Total']

        retail_column_config = {
            "Row Labels": st.column_config.Column(width=130),
            "Sum of Items Sold": st.column_config.Column(width=110),
            "Sum of Daily Sales": st.column_config.NumberColumn(  # 改为 NumberColumn
                "Sum of Daily Sales",
                width=130,
                format="%d"  # 去掉千位分隔符，直接显示数字
            ),
            "_sort_daily_sales": st.column_config.NumberColumn(
                "",
                width=1,
                format="%d"
            ),
            "daily_sales_raw": st.column_config.NumberColumn(width=1, label="", format="%d"),
            "Per day": st.column_config.Column(width=70),
            "Comments": st.column_config.Column(width=240),
            "Weekly change": st.column_config.NumberColumn(width=100, label="Weekly change", format="%.2f%%"),
        }

        # === CSS：强制两表等宽 ===
        st.markdown(f"""
        <style>
        .retail-table-wrapper {{
            width:{TABLE_WIDTH}px !important;
            max-width:{TABLE_WIDTH}px !important;
            margin: 0;
            padding: 0;
        }}
        .retail-table-wrapper [data-testid="stDataFrame"] {{
            width:{TABLE_WIDTH}px !important;
            max-width:{TABLE_WIDTH}px !important;
            min-width:{TABLE_WIDTH}px !important;
            overflow-x:hidden !important;
        }}
        .retail-table-wrapper [data-testid="stDataFrame"] table {{
            table-layout: fixed !important;
            width:{TABLE_WIDTH}px !important;
        }}
        .retail-table-wrapper [data-testid="stDataFrame"] td,
        .retail-table-wrapper [data-testid="stDataFrame"] th {{
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        # === 两表放同一容器，锁定一致宽度 ===
        with st.container():
            st.markdown("<div class='retail-table-wrapper'>", unsafe_allow_html=True)

            # --- Total表 ---
            st.dataframe(
                total_row_retail[["Row Labels", "Sum of Items Sold", "Sum of Daily Sales",
                                  "Per day", "Comments", "Weekly change"]],
                column_config=retail_column_config,
                hide_index=True,
                width=800
            )

            # === 添加：减少两个表格之间的间距 ===
            st.markdown("""
                    <style>
                    [data-testid="stDataFrame"] {
                        margin-top: -16px !important;
                        margin-bottom: -16px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

            # --- 主表 ---
            st.dataframe(
                other_rows_retail[["Row Labels", "Sum of Items Sold", "Sum of Daily Sales",
                                   "Per day", "Comments", "Weekly change"]],
                column_config=retail_column_config,
                hide_index=True,
                width=800
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # Retail分类商品项选择 - 使用与 high_level.py 相同的多选框样式
        st.markdown("<h4 style='font-size:16px; font-weight:700;'>📦 Retail Category Items</h4>", unsafe_allow_html=True)

        # 获取所有Retail分类的商品项
        retail_items_df = items_df[items_df["Category"].isin(retail_cats)].copy()
        if not retail_items_df.empty:
            # 使用新的清理函数移除所有计量单位
            retail_items_df["clean_item"] = retail_items_df["Item"].apply(clean_item_name_for_comments)
            retail_item_options = sorted(retail_items_df["clean_item"].dropna().unique())

            # 选择Retail分类和商品项 - 放在同一行
            col_retail1, col_retail2, col_retail3, _ = st.columns([1.2, 1.2, 1.6, 2.9])

            # --- Retail Categories ---
            with col_retail1:
                selected_retail_categories = persisting_multiselect_with_width(
                    "Select Retail Categories",
                    options=sorted(retail_df["Row Labels"].unique()),
                    key="retail_categories_select",
                    width_chars=22
                )

            # --- Search box ---
            with col_retail2:
                st.markdown("<div style='margin-top: 1.0rem;'></div>", unsafe_allow_html=True)
                retail_item_search_term = st.text_input(
                    "🔍 Search Items",
                    placeholder="Search keywords...",
                    key="retail_item_search_term"
                )

            with col_retail3:
                # ✅ 改进搜索逻辑：保留之前已选项
                if retail_item_search_term:
                    search_lower = retail_item_search_term.lower()
                    filtered_retail_items = [
                        item for item in retail_item_options if search_lower in str(item).lower()
                    ]

                    # ✅ 合并之前已选的项（防止输入新关键词后选项丢失）
                    prev_selected = st.session_state.get("retail_items_select", [])
                    filtered_retail_items = sorted(set(filtered_retail_items) | set(prev_selected))
                    item_count_text = f"{len(filtered_retail_items)} items (search active)"
                else:
                    filtered_retail_items = retail_item_options
                    item_count_text = f"{len(retail_item_options)} items"

                # 为商品项选择创建表单，避免立即 rerun
                with st.form(key="retail_items_form"):
                    # ✅ 改进：保留已选项，即使不在当前搜索结果中也能显示
                    current_selection = st.session_state.get("retail_items_select", [])
                    merged_options = sorted(set(filtered_retail_items) | set(current_selection))

                    selected_retail_items = st.multiselect(
                        f"Select Items ({item_count_text})",
                        options=merged_options,
                        default=current_selection,
                        key="retail_items_widget"
                    )

                    # 应用按钮
                    submitted_retail = st.form_submit_button("Apply", type="primary")

                    if submitted_retail:
                        # 更新 session state
                        st.session_state["retail_items_select"] = selected_retail_items
                        st.rerun()

                # 从 session state 获取最终的选择
                selected_retail_items = st.session_state.get("retail_items_select", [])

                # 显示当前选择状态（包括不在当前过滤列表中的选项）
                total_selected = len(selected_retail_items)
                if total_selected > 0:
                    visible_selected = len([item for item in selected_retail_items if item in filtered_retail_items])
                    if visible_selected == total_selected:
                        st.caption(f"✅ Selected: {total_selected} items")
                    else:
                        st.caption(f"✅ Selected: {total_selected} items ({visible_selected} visible)")
                else:
                    st.caption("ℹ️ No items selected")

            # 显示选中的商品项数据
            if selected_retail_categories or selected_retail_items:
                retail_item_summary = calculate_item_sales(
                    items_df, selected_retail_categories, selected_retail_items, start_date, end_date
                )

                if not retail_item_summary.empty:
                    # 设置列配置
                    item_column_config = {
                        'Category': st.column_config.Column(width="150px"),
                        'Item': st.column_config.Column(width="200px"),
                        'Sum of Items Sold': st.column_config.Column(width="130px"),
                        'Sum of Daily Sales': st.column_config.Column(width="100px")
                    }

                    st.data_editor(
                        retail_item_summary,
                        column_config=item_column_config,
                        use_container_width=False,
                        disabled=True,
                        hide_index=True
                    )

                    # 显示小计
                    total_qty = retail_item_summary["Sum of Items Sold"].sum()
                    total_sales = retail_item_summary["Sum of Daily Sales"].sum()
                    st.write(f"**Subtotal for selected items:** {total_qty} items, ${total_sales:,.0f}")


                    # === ✅ 与 Bar 部分完全一致的 Daily Trends 图表 ===
                    retail_daily_trends = calculate_item_daily_trends(
                        items_df, selected_retail_categories, selected_retail_items, start_date, end_date
                    )
                    retail_daily_trends["date_str"] = retail_daily_trends["date"].dt.strftime("%Y-%m-%d")

                    if not retail_daily_trends.empty:
                        # ✅ 多选框宽度，与 Select Retail Categories 一致
                        metric_col1, _ = st.columns([1.5, 5.5])
                        with metric_col1:
                            metric_option = persisting_multiselect_with_width(
                                label="Select metrics to display:",
                                options=["Sum of Items Sold", "Sum of Daily Sales"],
                                key="retail_daily_metric_select",
                                default=["Sum of Items Sold", "Sum of Daily Sales"],
                                width_chars=25  # 🔧 控制多选框宽度
                            )

                        # === 创建图形 ===
                        fig = go.Figure()

                        # --- 蓝色柱：Sum of Items Sold ---
                        if "Sum of Items Sold" in metric_option:
                            fig.add_trace(go.Bar(
                                x=retail_daily_trends["date_str"],
                                y=retail_daily_trends["Sum of Items Sold"],
                                name="Sum of Items Sold",
                                marker_color="#4F6D7A",
                                #width=0.3,
                                hovertemplate="Items Sold: %{y}<extra></extra>",
                                yaxis="y",  # 左轴
                                offsetgroup=0  # 分组编号 0
                            ))

                        # --- 红色柱：Sum of Daily Sales ---
                        if "Sum of Daily Sales" in metric_option:
                            fig.add_trace(go.Bar(
                                x=retail_daily_trends["date_str"],
                                y=retail_daily_trends["Sum of Daily Sales"],
                                name="Sum of Daily Sales ($)",
                                marker_color="#F2A65A",
                                #width=0.3,
                                hovertemplate="Sales: $%{y}<extra></extra>",
                                yaxis="y2",  # 右轴
                                offsetgroup=1  # 分组编号 1 → 不会重叠！
                            ))

                        show_items = "Sum of Items Sold" in metric_option
                        show_sales = "Sum of Daily Sales" in metric_option

                        # 网格逻辑：
                        # - 两个都选 → 网格来自左轴
                        # - 只选 Items Sold → 左轴网格
                        # - 只选 Daily Sales → 右轴网格
                        left_grid = show_items
                        right_grid = (show_sales and not show_items)

                        # === 更新布局 ===
                        fig.update_layout(
                            title="Daily Trends for Selected Items",
                            xaxis_title="Date",
                            hovermode="x unified",
                            # 左轴
                            yaxis=dict(
                                title="Items Sold",
                                showgrid=left_grid,
                                zeroline=True
                            ),

                            # 右轴
                            yaxis2=dict(
                                title="Sales ($)",
                                overlaying="y",
                                side="right",
                                showgrid=right_grid  # 当只有 sales 时自动开启
                            ),

                            barmode="group",  # 分组显示（两个柱并排）
                            bargap=0.02,
                            bargroupgap=0.02,
                            height=400,
                            margin=dict(t=60, b=60),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1,
                                font=dict(size=12, color="black", family="Arial")
                            )
                        )
                        fig.update_xaxes(tickformat="%b %d", tickangle=-45)

                        # ✅ 仅当只显示 Items Sold（未勾选 Sales）时，强制整数并自适配步长
                        if ("Sum of Items Sold" in metric_option) and ("Sum of Daily Sales" not in metric_option):
                            qty_max = int(max(1, (
                                retail_daily_trends if 'retail_daily_trends' in locals() else retail_daily_trends)[
                                "Sum of Items Sold"].max()))

                            # 目标 5~7 个刻度，选择 1/2/5×10^k 的“漂亮步长”
                            import math
                            def nice_dtick(n_max, target_ticks=6):
                                raw = max(1, math.ceil(n_max / target_ticks))
                                bases = [1, 2, 5]
                                k = 0
                                while True:
                                    for b in bases:
                                        step = b * (10 ** k)
                                        if step >= raw:
                                            return step
                                    k += 1

                            step = nice_dtick(qty_max)
                            fig.update_yaxes(
                                tickmode="linear",
                                tick0=0,
                                dtick=step,  # ← 关键：1/2/5×10^k 自适应
                                rangemode="tozero"  # 从 0 起，最大值交给 Plotly 自动算
                            )

                        # ✅ 图表居中显示，宽度为页面一半
                        chart_col1, _ = st.columns([1, 1])
                        with chart_col1:
                            st.plotly_chart(fig, config={"responsive": True, "displayModeBar": True})

                else:
                    st.info("No data for selected items.")
        else:
            st.info("No items found in Retail categories.")

    else:
        st.info("No data for Retail categories.")