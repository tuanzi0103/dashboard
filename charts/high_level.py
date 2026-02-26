import streamlit as st
import pandas as pd
import plotly.express as px
import math
import numpy as np
from services.db import get_db
from services.category_rules import is_bar_category

@st.cache_data(ttl=600)
def load_inventory_summary(db_mtime):
    db = get_db()
    df = pd.read_sql("SELECT * FROM inventory_summary", db)
    df["date"] = pd.to_datetime(df["date"])
    return df

def safe_fmt(x, digits=2, default="—"):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return f"{float(x):.{digits}f}"
    except Exception:
        return default

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


def proper_round(x):
    """标准的四舍五入方法，处理浮点数精度问题"""
    if pd.isna(x):
        return x
    # 处理浮点数精度问题
    x_rounded = round(x, 10)  # 先舍入到10位小数消除精度误差
    return math.floor(x_rounded + 0.5)


def persisting_multiselect(label, options, key, default=None, width_chars=None):
    if key not in st.session_state:
        st.session_state[key] = default or []

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


# === 预加载所有数据 ===


@st.cache_data(ttl=600, show_spinner=False)
def _prepare_inventory_grouped(inv: pd.DataFrame):
    if inv is None or inv.empty:
        return pd.DataFrame(), None

    df = inv.copy()

    if "source_date" in df.columns:
        df["date"] = pd.to_datetime(df["source_date"], errors="coerce")
        # === 修复：过滤掉转换失败的日期 ===
        df = df[df["date"].notna()]
    else:
        return pd.DataFrame(), None

    # Category 列
    if "Categories" in df.columns:
        df["Category"] = df["Categories"].astype(str)
    elif "Category" in df.columns:
        df["Category"] = df["Category"].astype(str)
    else:
        df["Category"] = "Unknown"

    # === 用 catalogue 现算 - 应用新的inventory value计算逻辑 ===
    # 1. 过滤掉 Current Quantity Vie Market & Bar 为负数或0的行
    df["Quantity"] = pd.to_numeric(df["Current Quantity Vie Market & Bar"], errors="coerce")
    mask = (df["Quantity"] > 0)  # 只保留正数
    df = df[mask].copy()

    if df.empty:
        return pd.DataFrame(), None

    tax_flag = df["Tax - GST (10%)"].astype(str)

    inventory_value = pd.Series(0.0, index=df.index)

    inventory_value.loc[tax_flag.eq("N")] = (
            df["UnitCost"] * df["Quantity"]
    ).loc[tax_flag.eq("N")]

    inventory_value.loc[tax_flag.eq("Y")] = (
            (df["UnitCost"] / 11.0 * 10.0) * df["Quantity"]
    ).loc[tax_flag.eq("Y")]

    df["Inventory Value"] = inventory_value.apply(proper_round)

    # 四舍五入保留整数
    df["Inventory Value"] = df["Inventory Value"].apply(lambda x: proper_round(x) if not pd.isna(x) else 0)

    # 保留其他计算（如果需要）
    df["Price"] = pd.to_numeric(df.get("Price", 0), errors="coerce").fillna(0)

    # 修复：检查 TaxFlag 列是否存在，如果不存在则创建默认值
    if "TaxFlag" not in df.columns:
        df["TaxFlag"] = "N"  # 默认值，假设不含税

    def calc_retail(row):
        try:
            O, AA, tax = row["Price"], row["Quantity"], row["TaxFlag"]
            return (O / 11 * 10) * AA if tax == "Y" else O * AA
        except KeyError:
            # 如果列不存在，直接计算 Price * Quantity
            return row["Price"] * row["Quantity"]

    df["Retail Total"] = df.apply(calc_retail, axis=1)
    df["Profit"] = df["Retail Total"] - df["Inventory Value"]

    # 聚合
    g = (
        df.groupby(["date", "Category"], as_index=False)[["Inventory Value", "Profit"]]
        .sum(min_count=1)
    )

    latest_date = g["date"].max() if not g.empty else None
    return g, latest_date

BAD_DATES = set(pd.to_datetime([
    "2025-08-18",
    "2025-08-19",
    "2025-08-20",
]))

@st.cache_data(ttl=600)
def load_summary():
    db = get_db()
    df = pd.read_sql("SELECT * FROM high_level_daily", db)
    df["date"] = pd.to_datetime(df["date"])
    return df

def show_high_level(tx: pd.DataFrame, mem: pd.DataFrame, inv: pd.DataFrame, bar_retail_data=None):
    from services.db import get_db_path
    import os
    db_path = get_db_path()
    db_mtime = os.path.getmtime(db_path)
    # === 全局样式：消除顶部标题间距 ===
    st.markdown("""
    <style>
    /* 去掉 Vie Manly Dashboard 与 High Level Report 之间的空白 */
    div.block-container h1, 
    div.block-container h2, 
    div.block-container h3, 
    div.block-container p {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    /* 更强力地压缩 Streamlit 自动插入的 vertical space */
    div.block-container > div {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    /* 消除标题和选择框之间空隙 */
    div[data-testid="stVerticalBlock"] > div {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # === 保留标题 ===
    st.markdown("<h2 style='font-size:24px; font-weight:700;'>📊 High Level Report</h2>", unsafe_allow_html=True)

    # 在现有的样式后面添加：
    st.markdown("""
    <style>
    /* 让多选框列更紧凑 */
    div[data-testid="column"] {
        padding: 0 8px !important;
    }
    div[data-baseweb="select"] {
        min-width: 12ch !important;
        max-width: 20ch !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 预加载所有数据
    with st.spinner("Loading data..."):
        summary_df = load_summary()
        inventory_summary_df = load_inventory_summary(db_mtime)

    # 初始化分类选择的 session state
    if "hl_cats" not in st.session_state:
        st.session_state["hl_cats"] = []
    if "hl_time" not in st.session_state:
        st.session_state["hl_time"] = ["MTD"]

    if "hl_data_base" not in st.session_state:
        st.session_state["hl_data_base"] = ["Daily Net Sales"]

    if "hl_cats" not in st.session_state or not st.session_state["hl_cats"]:
        st.session_state["hl_cats"] = ["total"]


    # === 特定日期选择 ===
    # 改为两列布局：时间范围选择 + 日期选择
    col_time_range, col_date, _ = st.columns([1, 1, 5])

    # === 添加空白行确保水平对齐 ===
    # st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <style>

    /* 让多选框列更紧凑 */
    div[data-testid="column"] {
        padding: 0 8px !important;
    }

    /* 精确控制 summary_time_range 下拉框宽度 */
    div[data-testid*="summary_time_range"] > div[data-baseweb="select"] {
        width: 14ch !important;
        min-width: 14ch !important;
        max-width: 14ch !important;
    }

    /* 日期选择框容器 - 精确宽度 */
    div[data-testid*="stSelectbox"] {
        width: 18ch !important;
        min-width: 18ch !important;
        max-width: 18ch !important;
        display: inline-block !important;
    }

    /* 日期选择框标签 */
    div[data-testid*="stSelectbox"] label {
        white-space: nowrap !important;
        font-size: 0.9rem !important;
        width: 100% !important;
    }

    /* 下拉菜单 */
    div[data-testid*="stSelectbox"] [data-baseweb="select"] {
        width: 18ch !important;
        min-width: 18ch !important;
        max-width: 18ch !important;
    }

    /* 下拉选项容器 */
    div[role="listbox"] {
        min-width: 18ch !important;
        max-width: 18ch !important;
    }

    /* 隐藏多余的下拉箭头空间 */
    div[data-testid*="stSelectbox"] [data-baseweb="select"] > div {
        padding-right: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with col_time_range:
        # === 移除空白标签，现在用CSS控制 ===
        summary_time_options = ["Daily", "WTD", "MTD", "YTD", "Custom date"]
        summary_time_range = st.selectbox(
            "Choose time range",
            summary_time_options,
            key="summary_time_range"
        )

    with col_date:
        # 直接从 summary 表获取日期
        available_dates = sorted(
            summary_df["date"].dropna().dt.date.unique(),
            reverse=True
        )

        if available_dates:
            available_dates_formatted = [
                d.strftime('%d/%m/%Y') for d in available_dates
            ]

            selected_date_formatted = st.selectbox(
                "Choose date",
                available_dates_formatted
            )

            selected_date = pd.to_datetime(
                selected_date_formatted,
                format='%d/%m/%Y'
            ).date()

        else:
            selected_date = pd.Timestamp.today().date()
            selected_date_formatted = selected_date.strftime('%d/%m/%Y')
            st.warning("No valid dates available.")

    # === 自定义日期范围选择（仅当选择Custom date时显示） ===
    summary_custom_dates_selected = False
    summary_t1 = None
    summary_t2 = None

    if summary_time_range == "Custom date":
        summary_custom_dates_selected = True
        st.markdown("<h4 style='font-size:16px; font-weight:700;'>📅 Custom Date Range for Summary</h4>",
                    unsafe_allow_html=True)

        col_from, col_to, _ = st.columns([1, 1, 5])

        with col_from:
            # ✅ 先确保 hl_date_from / hl_date_to 已经存在（建议在 show_high_level 更前面用 setdefault，见 3.3）
            summary_t1 = st.date_input(
                "From",
                value=st.session_state["hl_date_from"],
                key="hl_date_from",
                format="DD/MM/YYYY",
            )

        with col_to:
            summary_t2 = st.date_input(
                "To",
                value=st.session_state["hl_date_to"],
                key="hl_date_to",
                format="DD/MM/YYYY",
            )

    def filter_data_by_time_range(data, time_range, selected_date, custom_dates_selected=False, t1=None, t2=None):
        """根据时间范围筛选数据"""
        if data.empty:
            return data

        data_filtered = data.copy()

        # 获取当前日期
        today = pd.Timestamp.today().normalize()

        # 计算时间范围筛选条件
        start_of_week = today - pd.Timedelta(days=today.weekday())
        start_of_month = today.replace(day=1)
        start_of_year = today.replace(month=1, day=1)

        # 检查数据框是否有date列，如果没有则使用Datetime列
        if 'date' in data_filtered.columns:
            date_col = 'date'
        elif 'Datetime' in data_filtered.columns:
            date_col = 'Datetime'
            # 确保Datetime列是datetime类型
            data_filtered[date_col] = pd.to_datetime(data_filtered[date_col])
        else:
            # 如果没有日期列，返回原始数据
            return data_filtered

        # 确保日期列为 datetime 类型
        data_filtered[date_col] = pd.to_datetime(data_filtered[date_col], errors="coerce")

        # === 修复：优先处理 Custom date ===
        if custom_dates_selected and t1 and t2:
            t1_ts = pd.to_datetime(t1)
            t2_ts = pd.to_datetime(t2)
            data_filtered = data_filtered[
                (data_filtered[date_col] >= t1_ts) & (data_filtered[date_col] <= t2_ts)
                ]
        elif "WTD" in time_range:
            data_filtered = data_filtered[data_filtered[date_col] >= start_of_week]
        elif "MTD" in time_range:
            data_filtered = data_filtered[data_filtered[date_col] >= start_of_month]
        elif "YTD" in time_range:
            data_filtered = data_filtered[data_filtered[date_col] >= start_of_year]
        elif "Daily" in time_range:
            data_filtered = data_filtered[data_filtered[date_col].dt.date == selected_date]

        return data_filtered

    # 转换 selected_date 为 Timestamp 用于比较
    selected_date_ts = pd.Timestamp(selected_date)

    inv_filtered = inventory_summary_df[
        inventory_summary_df["date"] == pd.to_datetime(selected_date)
        ]

    inv_total = inv_filtered[inv_filtered["Category"] == "total"]

    if not inv_total.empty:
        inv_value_latest = float(inv_total.iloc[0]["inventory_value"])
        profit_latest = float(inv_total.iloc[0]["profit"])
    else:
        inv_value_latest = 0
        profit_latest = 0
    # ===== 直接从 summary 表读取 =====

    filtered = summary_df[
        summary_df["date"] == pd.to_datetime(selected_date)
        ]

    bar_row = filtered[filtered["Category"] == "bar"]
    retail_row = filtered[filtered["Category"] == "retail"]
    total_row = filtered[filtered["Category"] == "total"]

    if total_row.empty:
        st.warning("No data available.")
        return

    bar = bar_row.iloc[0] if not bar_row.empty else None
    retail = retail_row.iloc[0] if not retail_row.empty else None
    total = total_row.iloc[0]

    # 显示选定日期（字体加大）
    st.markdown(
        f"<h3 style='font-size:18px; font-weight:700;'>Selected Date: {selected_date.strftime('%d/%m/%Y')}</h3>",
        unsafe_allow_html=True)

    # === 计算 Bar / Retail 占比 ===
    total_net = total["daily_net_sales"] if total is not None else 0

    if total_net and total_net != 0:
        bar_pct = bar["daily_net_sales"] / total_net if bar is not None else 0
        retail_pct = retail["daily_net_sales"] / total_net if retail is not None else 0
    else:
        bar_pct = 0
        retail_pct = 0

    inv_filtered = inventory_summary_df[
        inventory_summary_df["date"] == pd.to_datetime(selected_date)
        ]

    inv_map = dict(
        zip(inv_filtered["Category"], inv_filtered["inventory_value"])
    )

    bar_inv = inv_map.get("bar", 0)
    retail_inv = inv_map.get("retail", 0)
    total_inv = inv_map.get("total", bar_inv + retail_inv)

    summary_data = {
        'Category': ['Bar', 'Retail', 'Total'],
        'Percentage': [
            f"{bar_pct:.1%}" if bar is not None else "-",
            f"{retail_pct:.1%}" if retail is not None else "-",
            "100%"
        ],

        'Daily Net Sales': [
            f"${proper_round(bar['daily_net_sales']):,}" if bar is not None else "-",
            f"${proper_round(retail['daily_net_sales']):,}" if retail is not None else "-",
            f"${proper_round(total['daily_net_sales']):,}"
        ],

        'Daily Transactions': [
            f"{proper_round(bar['transactions']):,}" if bar is not None else "-",
            f"{proper_round(retail['transactions']):,}" if retail is not None else "-",
            f"{proper_round(total['transactions']):,}"
        ],

        '# of Customers': [
            f"{proper_round(bar['customers']):,}" if bar is not None else "-",
            f"{proper_round(retail['customers']):,}" if retail is not None else "-",
            f"{proper_round(total['customers']):,}"
        ],

        'Avg Transaction': [
            f"${safe_fmt(bar['avg_txn'])}" if bar is not None else "-",
            f"${safe_fmt(retail['avg_txn'])}" if retail is not None else "-",
            f"${safe_fmt(total['avg_txn'])}"
        ],

        '3M Avg': [
            f"${proper_round(bar['rolling_90']):,}" if bar is not None else "-",
            f"${proper_round(retail['rolling_90']):,}" if retail is not None else "-",
            f"${proper_round(total['rolling_90']):,}"
        ],

        '6M Avg': [
            f"${proper_round(bar['rolling_180']):,}" if bar is not None else "-",
            f"${proper_round(retail['rolling_180']):,}" if retail is not None else "-",
            f"${proper_round(total['rolling_180']):,}"
        ],

        'Items Sold': [
            f"{proper_round(bar['qty']):,}" if bar is not None else "-",
            f"{proper_round(retail['qty']):,}" if retail is not None else "-",
            f"{proper_round(total['qty']):,}"
        ],

        "Inventory Value": [
            f"${proper_round(bar_inv):,}",
            f"${proper_round(retail_inv):,}",
            f"${proper_round(total_inv):,}",
        ],
    }
    df_summary = pd.DataFrame(summary_data)

    # ===== 渲染成 HTML 表格 =====
    # === 新增：Summary Table列宽配置 ===
    column_widths = {
        "label": "110px",
        "Percentage": "80px",
        "Daily Net Sales": "130px",
        "Daily Transactions": "140px",
        "# of Customers": "140px",
        "Avg Transaction": "125px",
        "3M Avg": "115px",
        "6M Avg": "115px",
        "Items Sold": "115px",
        "Inventory Value": "140px"
    }

    # 设置列配置
    column_config = {
        'Category': st.column_config.Column(width=80),
        'Percentage': st.column_config.Column(width=80),
        'Daily Net Sales': st.column_config.Column(width=100),
        'Daily Transactions': st.column_config.Column(width=120),
        '# of Customers': st.column_config.Column(width=100),
        'Avg Transaction': st.column_config.Column(width=105),
        '3M Avg': st.column_config.Column(width=55),
        '6M Avg': st.column_config.Column(width=55),
        'Items Sold': st.column_config.Column(width=75),
        'Inventory Value': st.column_config.Column(width=105),
    }
    # 显示表格
    st.markdown("<h4 style='font-size:16px; font-weight:700; margin-top:1rem;'>Summary Table</h4>",
                unsafe_allow_html=True)
    st.dataframe(
        df_summary,
        column_config=column_config,
        hide_index=True,
        width=875
    )

    st.markdown("---")

    # === 交互选择 ===
    st.markdown("<h4 style='font-size:16px; font-weight:700;'>🔍 Select Parameters</h4>", unsafe_allow_html=True)

    all_cats = sorted(summary_df["Category"].unique())

    priority = ["total", "retail", "bar"]

    # 先放优先级
    fixed_top = [c for c in priority if c in all_cats]

    # 再放其他
    others = [c for c in all_cats if c not in priority]

    all_cats_extended = fixed_top + others

    # === 四个多选框一行显示（使用 columns，等宽且靠左） ===

    # 定义每个框的宽度比例
    col1, col2, col3, col4, _ = st.columns([1.0, 1.2, 0.8, 1.5, 2.5])

    with col1:
        time_range = persisting_multiselect(
            "Choose time range",
            ["Custom date", "WTD", "MTD", "YTD"],
            key="hl_time",
            width_chars=15
        )


    with col2:
        data_sel_base = persisting_multiselect(
            "Choose data types",
            [
                "Daily Net Sales",
                "Weekly Net Sales",
                "Monthly Net Sales",  # ⭐ 新增
                "Daily Transactions",
                "Daily Number of Customers",  # ⭐ 新增
                "Avg Transaction",
                "Items Sold",
                "Inventory Value"
            ],
            key="hl_data_base",
            width_chars=22
        )

    with col3:
        data_sel_avg = persisting_multiselect(
            "Choose averages",
            ["3M Avg", "6M Avg"],
            key="hl_data_avg",
            width_chars=8
        )

    with col4:
        # 为分类选择创建表单，避免立即 rerun
        with st.form(key="categories_form"):
            cats_sel = st.multiselect(
                "Choose categories",
                all_cats_extended,
                default=st.session_state.get("hl_cats", []),
                key="hl_cats_widget"
            )

            # 应用按钮
            submitted = st.form_submit_button("Apply", type="primary")

            if submitted:
                st.session_state["hl_cats"] = cats_sel

        # 从 session state 获取最终的选择
        cats_sel = st.session_state.get("hl_cats", [])

        # 显示当前选择状态
        if cats_sel:
            st.caption(f"✅ Selected: {len(cats_sel)} categories")
        else:
            st.caption("ℹ️ No categories selected")

    # 加一小段 CSS，让四个框左对齐、间距最小
    st.markdown("""
    <style>
    div[data-testid="column"] {
        padding: 0 4px !important;
    }
    div[data-baseweb="select"] {
        min-width: 5ch !important;
        max-width: 35ch !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 合并数据类型选择
    data_sel = data_sel_base.copy()

    # 如果选择了平均值，为每个选择的基础数据类型添加对应的平均值
    for avg_type in data_sel_avg:
        for base_type in data_sel_base:
            if base_type in [
                "Daily Net Sales",
                "Weekly Net Sales",
                "Monthly Net Sales",  # ✅ 新增
                "Daily Transactions",
                "Daily Number of Customers",  # ✅ 新增
                "Avg Transaction",
                "Items Sold",
                # 如果你也想给库存做 avg，就加：
                "Inventory Value",  # ✅ 可选
                "Profit (Amount)"  # ✅ 可选
            ]:
                data_sel.append(f"{base_type} {avg_type}")


    # 如果没有选择任何基础数据类型但有平均值，默认使用Daily Net Sales
    if not data_sel_base and data_sel_avg:
        for avg_type in data_sel_avg:
            data_sel.append(f"Daily Net Sales {avg_type}")

    # === 自定义日期范围选择 ===
    custom_dates_selected = False

    st.session_state.setdefault("hl_date_from", pd.Timestamp.today().normalize() - pd.Timedelta(days=7))
    st.session_state.setdefault("hl_date_to", pd.Timestamp.today().normalize())

    t1 = st.session_state["hl_date_from"]
    t2 = st.session_state["hl_date_to"]

    if "Custom date" in time_range:
        custom_dates_selected = True
        st.markdown("<h4 style='font-size:16px; font-weight:700;'>📅 Custom Date Range</h4>", unsafe_allow_html=True)

        col_from, col_to, _ = st.columns([1, 1, 5])

        with col_from:
            # ✅ 不使用 key，直接获取返回值
            t1 = st.date_input(
                "From",
                value=st.session_state["hl_date_from"],
                key="hl_date_from",
                format="DD/MM/YYYY"
            )

        with col_to:
            # ✅ 不使用 key，直接获取返回值
            t2 = st.date_input(
                "To",
                value=st.session_state["hl_date_to"],
                key="hl_date_to",
                format="DD/MM/YYYY"
            )

    # 修改1：检查三个多选框是否都有选择
    has_time_range = bool(time_range)
    has_data_sel = bool(data_sel)
    has_cats_sel = bool(cats_sel)

    # 对于 Custom date，需要确保日期已选择
    if "Custom date" in time_range:
        has_valid_custom_dates = (t1 is not None and t2 is not None)
    else:
        has_valid_custom_dates = True

    # 实时计算图表数据 - 修改1：只有三个多选框都选择了才展示
    if has_time_range and has_data_sel and has_cats_sel and has_valid_custom_dates:
        with st.spinner("Generating chart..."):

            # === 修复：第一次进入 dashboard，Custom date 必须按用户选择生效 ===
            if "Custom date" in time_range:
                t1_final = t1
                t2_final = t2
            else:
                t1_final = None
                t2_final = None

            df_plot = filter_data_by_time_range(
                summary_df[summary_df["Category"].isin(cats_sel)],
                time_range,
                selected_date,
                custom_dates_selected,
                t1,
                t2
            )

        combined_list = []

        data_map = {

            # Sales
            "Daily Net Sales": "daily_net_sales",
            "Weekly Net Sales": "weekly_net_sales",
            "Monthly Net Sales": "monthly_net_sales",

            "Daily Net Sales 3M Avg": "rolling_90",
            "Daily Net Sales 6M Avg": "rolling_180",

            "Weekly Net Sales 3M Avg": "weekly_rolling_90",
            "Weekly Net Sales 6M Avg": "weekly_rolling_180",

            "Monthly Net Sales 3M Avg": "monthly_rolling_90",
            "Monthly Net Sales 6M Avg": "monthly_rolling_180",

            # Transactions
            "Daily Transactions": "transactions",
            "Daily Transactions 3M Avg": "transactions_rolling_90",
            "Daily Transactions 6M Avg": "transactions_rolling_180",

            # Customers
            "Daily Number of Customers": "customers",
            "Daily Number of Customers 3M Avg": "customers_rolling_90",
            "Daily Number of Customers 6M Avg": "customers_rolling_180",

            # Qty
            "Items Sold": "qty",
            "Items Sold 3M Avg": "qty_rolling_90",
            "Items Sold 6M Avg": "qty_rolling_180",

            # Avg txn
            "Avg Transaction": "avg_txn",
            "Avg Transaction 3M Avg": "avg_txn_rolling_90",
            "Avg Transaction 6M Avg": "avg_txn_rolling_180",

            # Inventory
            "Inventory Value": "inventory_value",
            "Inventory Value 3M Avg": "inventory_rolling_90",
            "Inventory Value 6M Avg": "inventory_rolling_180",
        }

        for dtype in data_sel:

            if dtype not in data_map:
                continue

            col = data_map[dtype]

            # 🔥 关键：自动判断数据来源
            if col.startswith("inventory"):
                source_df = inventory_summary_df
            else:
                source_df = df_plot

            temp_df = filter_data_by_time_range(
                source_df[source_df["Category"].isin(cats_sel)],
                time_range,
                selected_date,
                custom_dates_selected,
                t1,
                t2
            )

            if col not in temp_df.columns:
                continue

            temp = temp_df[["date", "Category", col]].copy()
            temp = temp.rename(columns={col: "value"})
            temp["data_type"] = dtype
            temp["series"] = temp["Category"] + " - " + dtype

            if col not in temp_df.columns:
                continue

            temp = temp_df[["date", "Category", col]].copy()
            temp = temp.rename(columns={col: "value"})
            temp["data_type"] = dtype
            temp["series"] = temp["Category"] + " - " + dtype

            combined_list.append(temp)

        if combined_list:
            combined_df = pd.concat(combined_list, ignore_index=True)
        else:
            combined_df = None

        if combined_df is not None and not combined_df.empty:
            # 修复：确保图表中的日期按正确顺序显示
            combined_df = combined_df.sort_values("date")

            # 立即显示图表
            fig = px.line(
                combined_df,
                x="date",
                y="value",
                color="series",
                title="All Selected Data Types by Category",
                labels={"date": "Date", "value": "Value", "series": "Series"}
            )

            # === 智能加 marker：只有一个点的 series 才加 marker ===
            series_counts = combined_df.groupby("series")["date"].nunique().to_dict()

            for trace in fig.data:
                name = trace.name
                if name in series_counts and series_counts[name] == 1:
                    trace.update(mode="markers", marker=dict(size=5))  # 只有一个点 → 放大显示
                else:
                    trace.update(mode="lines")  # 正常多点 → 保持线图

            # 改为欧洲日期格式
            fig.update_layout(
                xaxis=dict(tickformat="%d/%m/%Y"),
                hovermode="x unified",
                height=600
            )

            # ✅ 强制 X 轴显示完整自定义日期范围（避免 Plotly 自动缩放只显示末段）
            if "Custom date" in time_range and t1_final is not None and t2_final is not None:
                t1_ts = pd.to_datetime(t1_final)
                t2_ts = pd.to_datetime(t2_final)
                week_start = t1_ts - pd.Timedelta(days=t1_ts.weekday())  # 回到周一
                fig.update_xaxes(range=[week_start, t2_ts])

            st.plotly_chart(
                fig,
                config={
                    "responsive": True,
                    "displayModeBar": True
                }
            )

            st.markdown("""
            <style>
            div[data-testid="stExpander"] > div:first-child {
                width: fit-content !important;
                max-width: 95% !important;
            }
            div[data-testid="stDataFrame"] {
                width: fit-content !important;
            }
            </style>
            """, unsafe_allow_html=True)

            # 显示数据表格 - 直接展示，去掉下拉框
            st.markdown("#### 📊 Combined Data for All Selected Types")
            display_df = combined_df.copy()

            # === 修改：为 Weekly Net Sales 显示周区间 ===
            def format_weekly_date(row):
                if "Weekly Net Sales" in row["data_type"]:
                    # 计算周的起始和结束日期（周一到周日）
                    week_start = row["date"]
                    week_end = week_start + pd.Timedelta(days=6)
                    # 确保周区间不重叠：如果起始日期不是周一，调整为周一
                    if week_start.weekday() != 0:  # 0 代表周一
                        week_start = week_start - pd.Timedelta(days=week_start.weekday())
                        week_end = week_start + pd.Timedelta(days=6)
                    return f"{week_start.strftime('%d/%m/%Y')}-{week_end.strftime('%d/%m/%Y')}"
                else:
                    return row["date"].strftime("%d/%m/%Y")

            display_df["date"] = display_df.apply(format_weekly_date, axis=1)

            # === 修改：对表格中的 Daily Net Sales 和 Weekly Net Sales 也进行四舍五入取整 ===
            display_df.loc[display_df["data_type"].isin(["Daily Net Sales", "Weekly Net Sales"]), "value"] = \
                display_df.loc[
                    display_df["data_type"].isin(["Daily Net Sales", "Weekly Net Sales"]), "value"
                ].apply(lambda x: proper_round(x) if not pd.isna(x) else 0)

            display_df = display_df.rename(columns={
                "date": "Date",
                "Category": "Category",
                "data_type": "Data Type",
                "value": "Value"
            })

            # 修复：按日期正确排序（需要创建一个临时日期列用于排序）
            def get_sort_date(row):
                if "Weekly Net Sales" in row["Data Type"]:
                    # 从周区间中提取起始日期
                    start_date_str = row["Date"].split('-')[0]
                    return pd.to_datetime(start_date_str, format='%d/%m/%Y')
                else:
                    return pd.to_datetime(row["Date"], format='%d/%m/%Y')

            display_df["Date_dt"] = display_df.apply(get_sort_date, axis=1)
            display_df = display_df.sort_values(["Date_dt", "Category", "Data Type"])
            display_df = display_df.drop("Date_dt", axis=1)

            # === 修改1：表格容器宽度跟随表格内容 ===
            # 计算表格总宽度
            total_width = 0
            for column in display_df.columns:
                header_len = len(str(column))
                # 估算列宽：标题长度+数据最大长度+2字符边距
                data_len = display_df[column].astype(str).str.len().max()
                col_width = max(header_len, data_len) + 2
                total_width += col_width

            # 设置表格容器样式
            st.markdown(f"""
            <style>
            /* 表格容器 - 宽度跟随内容 */
            [data-testid="stExpander"] {{
                width: auto !important;
                min-width: {total_width}ch !important;
                max-width: 100% !important;
            }}
            /* 让表格左右可滚动 */
            [data-testid="stDataFrame"] div[role="grid"] {{
                overflow-x: auto !important;
                width: auto !important;
            }}
            /* 自动列宽，不强制占满 */
            [data-testid="stDataFrame"] table {{
                table-layout: auto !important;
                width: auto !important;
            }}
            /* 所有单元格左对齐 */
            [data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {{
                text-align: left !important;
                justify-content: flex-start !important;
            }}
            /* 防止省略号 */
            [data-testid="stDataFrame"] td {{
                white-space: nowrap !important;
            }}
            </style>
            """, unsafe_allow_html=True)

            # === 新逻辑：列宽根据标题字符串长度设置 ===
            column_config = {}
            for column in display_df.columns:
                header_len = len(str(column))
                column_config[column] = st.column_config.Column(
                    column,
                    width=f"{header_len + 2}ch"
                )

            # 对3M/6M平均值列四舍五入保留两位小数
            avg_mask = display_df["Data Type"].str.contains("3M Avg|6M Avg", case=False, na=False)
            display_df.loc[avg_mask, "Value"] = display_df.loc[avg_mask, "Value"].apply(
                lambda x: round(x, 2) if pd.notna(x) else x
            )

            # 新增：对 Weekly Net Sales 也进行四舍五入取整
            weekly_mask = display_df["Data Type"].str.contains("Weekly Net Sales", case=False, na=False) & ~display_df[
                "Data Type"].str.contains("Avg", case=False, na=False)
            display_df.loc[weekly_mask, "Value"] = display_df.loc[weekly_mask, "Value"].apply(
                lambda x: proper_round(x) if not pd.isna(x) else 0
            )

            st.dataframe(display_df, column_config=column_config)

        else:
            st.warning("No data available for the selected combination.")

