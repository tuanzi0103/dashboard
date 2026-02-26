import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Optional

from services.analytics import (
    forecast_top_consumers,
    sku_consumption_timeseries,
)
from services.simulator import simulate_consumption, simulate_consumption_timeseries


def detect_store_current_qty_col(df_inv: pd.DataFrame) -> Optional[str]:
    if df_inv is None or df_inv.empty:
        return None
    norm = {c: str(c).lower().strip() for c in df_inv.columns}
    for c, n in norm.items():
        if n.startswith("current quantity"):
            return c
    return None


def persisting_multiselect(label, options, key, default=None, width_chars=None):
    """
    保持选择状态的多选框函数 - 统一宽度和箭头显示（增强版）
    """
    if key not in st.session_state:
        st.session_state[key] = default or []

    if width_chars is None:
        min_width = 30  # 全局默认 30ch
    else:
        min_width = width_chars

    st.markdown(f"""
    <style>
    /* === 强制覆盖 stMultiSelect 宽度（仅限当前 key） === */
    div[data-testid="stMultiSelect"][data-testid*="{key}"],
    [data-testid*="{key}"][data-testid="stMultiSelect"] {{
        width: {min_width}ch !important;
        min-width: {min_width}ch !important;
        max-width: {min_width}ch !important;
        flex: 0 0 {min_width}ch !important;
        box-sizing: border-box !important;
    }}

    /* === 下拉框主体 === */
    div[data-testid="stMultiSelect"][data-testid*="{key}"] [data-baseweb="select"],
    div[data-testid="stMultiSelect"][data-testid*="{key}"] [data-baseweb="select"] > div {{
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }}

    /* === 输入框 === */
    div[data-testid="stMultiSelect"][data-testid*="{key}"] input {{
        width: 100% !important;
        box-sizing: border-box !important;
    }}

    /* === 下拉菜单 === */
    div[role="listbox"] {{
        width: {min_width}ch !important;
        min-width: {min_width}ch !important;
        max-width: {min_width}ch !important;
        box-sizing: border-box !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # ✅ 新逻辑：保留之前已选项（不会因为搜索过滤被清空）
    prev_selected = st.session_state[key]
    merged_options = sorted(set(options) | set(prev_selected))  # 合并已选项 + 当前过滤结果

    selected = st.multiselect(label, merged_options, default=prev_selected, key=key)
    return selected


def filter_by_time_range(df, time_range, custom_dates_selected=False, t1=None, t2=None):
    """根据时间范围筛选数据"""
    if df is None or df.empty:
        return df

    # 如果没有日期列，直接返回原数据
    if "date" not in df.columns and "source_date" not in df.columns:
        return df

    # 获取日期列名
    date_col = "date" if "date" in df.columns else "source_date"

    # 确保日期列是datetime类型
    df_filtered = df.copy()
    df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors="coerce")

    # 获取当前日期
    today = pd.Timestamp.today().normalize()

    # 计算时间范围
    start_of_week = today - pd.Timedelta(days=today.weekday())
    start_of_month = today.replace(day=1)
    start_of_year = today.replace(month=1, day=1)

    # 应用时间范围筛选 - 这里要使用 date_col 变量而不是硬编码的 "date"
    if "WTD" in time_range:
        df_filtered = df_filtered[df_filtered[date_col] >= start_of_week]
    if "MTD" in time_range:
        df_filtered = df_filtered[df_filtered[date_col] >= start_of_month]
    if "YTD" in time_range:
        df_filtered = df_filtered[df_filtered[date_col] >= start_of_year]
    if custom_dates_selected and t1 and t2:
        t1_ts = pd.to_datetime(t1)
        t2_ts = pd.to_datetime(t2)
        df_filtered = df_filtered[
            (df_filtered[date_col] >= t1_ts) & (df_filtered[date_col] <= t2_ts)
            ]

    return df_filtered


def calculate_inventory_summary(inv_df):
    """计算库存汇总数据"""
    if inv_df is None or inv_df.empty:
        return {
            "Total Inventory Value": 0,
            "Total Retail Value": 0,
            "Profit": 0,
            "Profit Margin": "0.0%"
        }

    df = inv_df.copy()

    # 1. 过滤掉负数、0、空值的库存和成本
    df["Quantity"] = pd.to_numeric(df["Current Quantity Vie Market & Bar"], errors="coerce")
    df["UnitCost"] = pd.to_numeric(df["Default Unit Cost"], errors="coerce")
    df = df[(df["Quantity"] > 0) & (df["UnitCost"] > 0)].copy()

    if df.empty:
        return {
            "Total Inventory Value": 0,
            "Total Retail Value": 0,
            "Profit": 0,
            "Profit Margin": "0.0%"
        }

    # 2. 处理单位成本
    df["UnitCost"] = pd.to_numeric(df["Default Unit Cost"], errors="coerce").fillna(0)

    # 3. 计算 Inventory Value
    df["Inventory Value"] = df["UnitCost"] * df["Quantity"]
    total_inventory_value = df["Inventory Value"].sum()

    # 4. 计算 Total Retail Value
    def calc_single_retail(row):
        try:
            O, AA, tax = row["Price"], row["Quantity"], str(row["Tax - GST (10%)"]).strip().upper()
            return (O / 11 * 10) * AA if tax == "Y" else O * AA
        except KeyError:
            return row["Price"] * row["Quantity"]

    df["Single Retail Value"] = df.apply(calc_single_retail, axis=1)
    total_retail_value = df["Single Retail Value"].sum()

    # 5. 计算 Profit 和 Profit Margin
    profit = total_retail_value - total_inventory_value
    profit_margin = (profit / total_retail_value * 100) if total_retail_value > 0 else 0

    # 四舍五入
    total_inventory_value = round(total_inventory_value)
    total_retail_value = round(total_retail_value)
    profit = round(profit)
    total_inventory_value = int(total_inventory_value)

    return {
        "Total Inventory Value": total_inventory_value,
        "Total Retail Value": total_retail_value,
        "Profit": profit,
        "Profit Margin": f"{profit_margin:.1f}%"
    }


def show_inventory(tx, inventory: pd.DataFrame):
    # === 全局样式：参考 high_level 的样式设置 ===
    st.markdown("""
    <style>
    /* 去掉标题之间的空白 */
    div.block-container h1, 
    div.block-container h2, 
    div.block-container h3, 
    div.block-container h4,
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

    /* 让多选框列更紧凑 */
    div[data-testid="column"] {
        padding: 0 8px !important;
    }
    /* 让表格文字左对齐 */
    [data-testid="stDataFrame"] table {
        text-align: left !important;
    }
    [data-testid="stDataFrame"] th {
        text-align: left !important;
    }
    [data-testid="stDataFrame"] td {
        text-align: left !important;
    }

    /* 让 Current Quantity 输入框和多选框对齐 */
    div[data-testid*="stNumberInput"] {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    div[data-testid*="stNumberInput"] label {
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }

    /* 统一多选框和输入框的垂直对齐 */
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        align-items: start !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    /* 让 data_editor 不自动撑开列宽 */
    [data-testid="stDataEditor"] table {
        table-layout: fixed !important;
    }

    /* 内容超出列宽 → 自动省略号，不撑开 */
    [data-testid="stDataEditor"] td, 
    [data-testid="stDataEditor"] th {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # === 标题样式参考 high_level ===
    st.markdown("<h2 style='font-size:24px; font-weight:700;'>📦 Product Mix & Inventory Optimization</h2>",
                unsafe_allow_html=True)

    if tx.empty:
        st.info("No transaction data available")
        return

    if inventory is None or inventory.empty:
        st.info("No inventory data available")
        return

    # === 新增：统一处理带星号的 item 名称 ===
    tx = tx.copy()
    if "Item" in tx.columns:
        # 移除 Item 列开头的星号
        tx["Item"] = tx["Item"].astype(str).str.replace(r'^\*', '', regex=True).str.strip()

    # 同样处理其他可能的 item 相关列
    if "Item Name" in tx.columns:
        tx["Item Name"] = tx["Item Name"].astype(str).str.replace(r'^\*', '', regex=True).str.strip()

    if "Price Point Name" in tx.columns:
        tx["Price Point Name"] = tx["Price Point Name"].astype(str).str.replace(r'^\*', '', regex=True).str.strip()

    inv = inventory.copy()

    # ---- 💰 Inventory Valuation Analysis ----
    st.markdown("<h3 style='font-size:20px; font-weight:700;'>💰 Inventory Valuation Analysis</h3>",
                unsafe_allow_html=True)

    # === 修改：只保留日期选择框 ===
    col_date, _, _, _ = st.columns([1, 1, 1.8, 3.5])

    with col_date:
        # 获取可用的日期（从库存数据中提取）
        if "source_date" in inv.columns:
            available_dates = sorted(pd.to_datetime(inv["source_date"]).dt.date.unique(), reverse=True)
        elif "date" in inv.columns:
            available_dates = sorted(pd.to_datetime(inv["date"]).dt.date.unique(), reverse=True)
        else:
            available_dates = []

        # 将日期格式改为欧洲格式显示
        available_dates_formatted = [date.strftime('%d/%m/%Y') for date in available_dates]

        # === 修复：使用正确的 CSS 选择器设置日期选择框宽度 ===
        st.markdown("""
        <style>
        /* 仅影响日期选择框：通过label名称或key限定 */
        div[data-testid*="stSelectbox"][aria-label="Choose date"],
        div[data-testid*="stSelectbox"][data-baseweb="select"][aria-label="Choose date"] {
            width: 18ch !important;
            min-width: 18ch !important;
            max-width: 18ch !important;
        }
        </style>
        """, unsafe_allow_html=True)

        selected_date_formatted = st.selectbox("Choose date", available_dates_formatted)

        # 将选择的日期转换回日期对象
        selected_date = pd.to_datetime(selected_date_formatted, format='%d/%m/%Y').date()

    # 转换 selected_date 为 Timestamp 用于比较
    selected_date_ts = pd.Timestamp(selected_date)

    # 移除原有的时间范围选择逻辑，现在使用单一日期
    time_range = []  # 清空时间范围，因为现在只用单一日期
    custom_dates_selected = False
    t1 = None
    t2 = None

    # ---- Inventory Summary Table ----
    # 获取选定日期的库存数据
    if "source_date" in inv.columns or "date" in inv.columns:
        date_col = "source_date" if "source_date" in inv.columns else "date"
        inv_with_date = inv.copy()
        inv_with_date[date_col] = pd.to_datetime(inv_with_date[date_col], errors="coerce")
        # 筛选选定日期的数据
        filtered_inv = inv_with_date[inv_with_date[date_col].dt.date == selected_date]
        summary_data = calculate_inventory_summary(filtered_inv)
    else:
        summary_data = calculate_inventory_summary(inv)

    # 显示选定日期 - 参考 high_level 的格式
    st.markdown(
        f"<h4 style='font-size:16px; font-weight:700;'>Selected Date: {selected_date.strftime('%d/%m/%Y')}</h4>",
        unsafe_allow_html=True)

    # === 修改：Selected Date 横向展示 ===
    summary_table_data = {
        'Total Inventory Value': [f"${summary_data['Total Inventory Value']:,}"],
        'Total Retail Value': [f"${summary_data['Total Retail Value']:,}"],
        'Profit': [f"${summary_data['Profit']:,}"],
        'Profit Margin': [summary_data['Profit Margin']]
    }
    df_summary = pd.DataFrame(summary_table_data)

    column_config = {
        'Total Inventory Value': st.column_config.Column(width=140),
        'Total Retail Value': st.column_config.Column(width=110),
        'Profit': st.column_config.Column(width=60),
        'Profit Margin': st.column_config.Column(width=90),
    }

    st.data_editor(
        df_summary,
        column_config=column_config,
        use_container_width=False,
        hide_index=True,
        disabled=True
    )

    st.markdown("---")

    # === 修改：将Low Stock Alerts的内容移动到Summary Table下面 ===

    # ---- Low Stock Alerts ----
    # === 生成低库存表 ===
    low_stock = filtered_inv.copy()
    qty_col = detect_store_current_qty_col(inv)

    # ✅ 确保存在 option_key 列
    if "option_key" not in low_stock.columns:
        item_col = "Item Name" if "Item Name" in low_stock.columns else "Item"
        variation_col = "Variation Name" if "Variation Name" in low_stock.columns else None
        sku_col = "SKU" if "SKU" in low_stock.columns else None

        if variation_col:
            low_stock["display_name"] = low_stock[item_col].astype(str) + " - " + low_stock[variation_col].astype(str)
        else:
            low_stock["display_name"] = low_stock[item_col].astype(str)

        if sku_col:
            low_stock["option_key"] = low_stock["display_name"] + " (SKU:" + low_stock[sku_col].astype(str) + ")"
        else:
            low_stock["option_key"] = low_stock["display_name"]

    # 修改后：
    low_stock = low_stock[pd.notna(pd.to_numeric(low_stock[qty_col], errors="coerce"))].copy()
    if not low_stock.empty:
        options = sorted(low_stock["option_key"].unique())

        # === 修改：参考 Inventory Valuation Analysis 的布局，使用五列布局 ===
        col_search_low, col_select_low, col_threshold_low, col_threshold_high, _ = st.columns([1, 1.8, 1, 1, 2.2])

        with col_search_low:
            st.markdown("<div style='margin-top: 1.0rem;'></div>", unsafe_allow_html=True)
            # === 修改：添加二级搜索框 ===
            low_stock_search_term = st.text_input(
                "🔍 Search",
                placeholder="Search items...",
                key="low_stock_search_term"
            )

        with col_select_low:
            # 根据搜索词过滤选项
            if low_stock_search_term:
                search_lower = low_stock_search_term.lower()
                filtered_options = [item for item in options if search_lower in str(item).lower()]
                prev_selected = st.session_state.get("low_stock_filter", [])
                filtered_options = sorted(set(filtered_options) | set(prev_selected))
                item_count_text = f"{len(filtered_options)} items"
            else:
                filtered_options = options
                item_count_text = f"{len(options)} items"

            # === 用 form 包裹，防止选择时自动 rerun ===
            with st.form(key="low_stock_form"):
                selected_temp = st.multiselect(
                    f"Select Items ({item_count_text})",
                    filtered_options,
                    default=st.session_state.get("low_stock_filter", []),
                    key="low_stock_filter_temp"
                )

                # 红色 Apply 按钮样式（和 high_level 一样）
                st.markdown("""
                <style>
                div[data-testid="stFormSubmitButton"] button {
                    background-color: #ff4b4b !important;
                    color: white !important;
                    font-weight: 600 !important;
                    border: none !important;
                    border-radius: 8px !important;
                    height: 2.2em !important;
                    width: 100% !important;
                }
                </style>
                """, unsafe_allow_html=True)

                submitted = st.form_submit_button("Apply")

                if submitted:
                    st.session_state["low_stock_filter"] = selected_temp
                    st.success("Selections applied!")

            # 从 session_state 获取最终选择
            selected_items = st.session_state.get("low_stock_filter", [])

        with col_threshold_low:
            # Current Quantity ≤
            st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

            # === 修改：改为单选框，直接输入数字作为阈值 ===
            max_qty = int(low_stock[qty_col].max())
            threshold_low_value = st.number_input(
                "Current Quantity ≤",
                min_value=1,
                max_value=20,
                value=20,
                key="low_stock_threshold_low",
                help="Enter threshold value for low stock"
            )

        with col_threshold_high:
            # === 新增：Current Quantity ≥ 多选框 ===
            st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)

            threshold_high_value = st.number_input(
                "Current Quantity ≥",
                min_value=0,
                max_value=100,
                value=0,
                key="low_stock_threshold_high",
                help="Enter threshold value for high stock"
            )

        df_low = low_stock.copy()
        df_low["current_qty"] = pd.to_numeric(df_low[qty_col], errors="coerce").fillna(0)

        if selected_items:
            selected_skus = [opt.split("SKU:")[1].replace(")", "") for opt in selected_items if "SKU:" in opt]
            if selected_skus:
                df_low = df_low[df_low["SKU"].astype(str).isin(selected_skus)]
            else:
                df_low = df_low[df_low["display_name"].isin(selected_items)]

        if not df_low.empty:
            df_low_display = df_low.copy()

            # 应用阈值筛选：同时应用 ≤ 和 ≥ 条件
            current_qty_numeric = pd.to_numeric(df_low_display[qty_col], errors="coerce").fillna(0)

            # ==== Default 开关 ====
            no_qty_filter = st.checkbox(
                "Show ALL Items (Ignore Current Quantity Filters)",
                key="no_qty_filter",
            )

            # 数字化库存
            current_qty_numeric = pd.to_numeric(df_low_display[qty_col], errors="coerce").fillna(0)

            # ==== 修改后的过滤逻辑 ====
            # 如果用户没有勾选 default → 才执行库存过滤
            if not no_qty_filter:

                # ≤ 条件
                if threshold_low_value > 0:
                    df_low_display = df_low_display[current_qty_numeric <= threshold_low_value]

                # ≥ 条件
                if threshold_high_value > 0:
                    df_low_display = df_low_display[current_qty_numeric >= threshold_high_value]

            # 如果 no_qty_filter=True → 自动跳过过滤（不执行任何限制）

            # 确保数值列是数字类型
            df_low_display["Current Quantity Vie Market & Bar"] = pd.to_numeric(
                df_low_display["Current Quantity Vie Market & Bar"], errors="coerce").fillna(0)
            df_low_display["Price"] = pd.to_numeric(df_low_display["Price"], errors="coerce").fillna(0)
            df_low_display["Default Unit Cost"] = pd.to_numeric(df_low_display["Default Unit Cost"],
                                                                errors="coerce").fillna(0)

            # 计算 Total Inventory (使用绝对值)
            df_low_display["Total Inventory"] = df_low_display["Default Unit Cost"] * abs(
                df_low_display["Current Quantity Vie Market & Bar"])

            # 计算 Total Retail
            def calc_retail(row):
                O, AA, tax = row["Price"], abs(row["Current Quantity Vie Market & Bar"]), str(
                    row["Tax - GST (10%)"]).strip().upper()
                return (O / 11 * 10) * AA if tax == "Y" else O * AA

            df_low_display["Total Retail"] = df_low_display.apply(calc_retail, axis=1)

            # 计算 Profit
            df_low_display["Profit"] = df_low_display["Total Retail"] - df_low_display["Total Inventory"]

            # 所有数值列先四舍五入处理浮点数精度问题
            df_low_display["Total Inventory"] = df_low_display["Total Inventory"].round(2)
            df_low_display["Total Retail"] = df_low_display["Total Retail"].round(2)
            df_low_display["Profit"] = df_low_display["Profit"].round(2)

            # === 修改：Profit Margin 始终计算，即使没有库存或总值 ===
            def calc_profit_margin(row):
                try:
                    unit_cost = float(row.get("Default Unit Cost", 0))
                    price = float(row.get("Price", 0))
                    tax = str(row.get("Tax - GST (10%)", "")).strip().upper()

                    # 按单价计算，不依赖数量或总额
                    effective_price = (price / 11 * 10) if tax == "Y" else price
                    if effective_price == 0:
                        return "0.0%"
                    profit = effective_price - unit_cost
                    profit_margin = (profit / effective_price) * 100
                    return f"{profit_margin:.1f}%"
                except Exception:
                    return "0.0%"

            df_low_display["Profit Margin"] = df_low_display.apply(calc_profit_margin, axis=1)

            # 计算过去4周的Net Sales
            #selected_date_ts = pd.Timestamp(selected_date)

            selected_date_ts = pd.Timestamp(selected_date).normalize()
            end_ts = selected_date_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

            # === 修改：按 Item Name 和 Variation Name 连接 transaction 表 ===
            tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")
            past_4w_start = selected_date_ts - pd.Timedelta(days=28)
            recent_tx = tx[(tx["Datetime"] >= past_4w_start) & (tx["Datetime"] <= end_ts)].copy()

            recent_tx["Item"] = recent_tx["Item"].astype(str).str.strip()
            # 移除 Item 列开头的星号（确保与前面处理一致）
            recent_tx["Item"] = recent_tx["Item"].str.replace(r'^\*', '', regex=True).str.strip()
            recent_tx["Price Point Name"] = recent_tx["Price Point Name"].astype(str).str.strip()
            recent_tx["Net Sales"] = pd.to_numeric(recent_tx["Net Sales"], errors="coerce").fillna(0)

            # 按 Item Name 和 Price Point Name 分组计算销售额
            item_sales_4w = (
                recent_tx.groupby(["Item", "Price Point Name"])["Net Sales"]
                .sum()
                .reset_index()
                .rename(columns={"Item": "Item Name", "Price Point Name": "Variation Name",
                                 "Net Sales": "Net Sale 4W"})
            )

            # === 新增：计算过去3个月和6个月（自然月份） ===

            past_3m_start = selected_date_ts - pd.Timedelta(days=90)
            past_6m_start = selected_date_ts - pd.Timedelta(days=180)

            # ---- Last 3 Months ----
            tx_3m = tx[(tx["Datetime"] >= past_3m_start) & (tx["Datetime"] <= end_ts)].copy()
            tx_3m["Net Sales"] = pd.to_numeric(tx_3m["Net Sales"], errors="coerce").fillna(0)
            tx_3m["Item"] = tx_3m["Item"].astype(str).str.strip().str.replace(r'^\*', '', regex=True)
            tx_3m["Price Point Name"] = tx_3m["Price Point Name"].astype(str).str.strip()

            item_sales_3m = (
                tx_3m.groupby(["Item", "Price Point Name"])["Net Sales"]
                .sum()
                .reset_index()
                .rename(columns={
                    "Item": "Item Name",
                    "Price Point Name": "Variation Name",
                    "Net Sales": "Last 3 Months Sales"
                })
            )

            # ---- Last 6 Months ----
            tx_6m = tx[(tx["Datetime"] >= past_6m_start) & (tx["Datetime"] <= end_ts)].copy()
            tx_6m["Net Sales"] = pd.to_numeric(tx_6m["Net Sales"], errors="coerce").fillna(0)
            tx_6m["Item"] = tx_6m["Item"].astype(str).str.strip().str.replace(r'^\*', '', regex=True)
            tx_6m["Price Point Name"] = tx_6m["Price Point Name"].astype(str).str.strip()

            item_sales_6m = (
                tx_6m.groupby(["Item", "Price Point Name"])["Net Sales"]
                .sum()
                .reset_index()
                .rename(columns={
                    "Item": "Item Name",
                    "Price Point Name": "Variation Name",
                    "Net Sales": "Last 6 Months Sales"
                })
            )

            def smart_merge(df_inv, df_tx):
                """
                修复 6M < 3M 的根本问题：
                1. 当 Variation Name 不一致时，优先按 Item 匹配
                2. merge 时不覆盖已有数据（正确累加）
                3. 允许 Variation 为 nan/空字符串/不一致
                """

                df_inv2 = df_inv.copy()
                df_tx2 = df_tx.copy()

                # 标准化 Variation Name
                df_inv2["Variation Name"] = df_inv2["Variation Name"].fillna("").astype(str).str.strip()
                df_tx2["Variation Name"] = df_tx2["Variation Name"].fillna("").astype(str).str.strip()

                # 先按 (Item Name + Variation Name) 精确匹配
                merged = df_inv2.merge(df_tx2, on=["Item Name", "Variation Name"], how="left")

                # 退回用 Item Name 仅匹配（宽松匹配）
                fallback = df_inv2.merge(
                    df_tx2.groupby("Item Name").sum(numeric_only=True).reset_index(),
                    on="Item Name",
                    how="left"
                )

                # combine_first：已有的非空值优先，不被错误覆盖
                for col in df_tx2.columns:
                    if col not in ["Item Name", "Variation Name"]:
                        merged[col] = merged[col].combine_first(fallback[col])

                return merged

            df_low_display = smart_merge(df_low_display, item_sales_4w)
            df_low_display = smart_merge(df_low_display, item_sales_3m)
            df_low_display = smart_merge(df_low_display, item_sales_6m)

            df_low_display["Velocity"] = df_low_display.apply(
                lambda r: round(r["Total Retail"] / r["Net Sale 4W"], 2)
                if pd.notna(r["Net Sale 4W"]) and r["Net Sale 4W"] > 0
                else "-",
                axis=1
            )

            # Velocity 四舍五入保留一位小数
            vel_numeric = pd.to_numeric(df_low_display["Velocity"], errors="coerce")
            df_low_display["Velocity"] = vel_numeric.round(1).where(vel_numeric.notna(), df_low_display["Velocity"])

            # 重命名 Current Quantity Vie Market & Bar 列为 Current Quantity
            df_low_display = df_low_display.rename(columns={"Current Quantity Vie Market & Bar": "Current Quantity"})

            # 选择要显示的列 - 在 Item Name 右边添加 Variation Name
            display_columns = []
            if "Item Name" in df_low_display.columns:
                display_columns.append("Item Name")
            if "Variation Name" in df_low_display.columns:
                display_columns.append("Variation Name")

            display_columns.extend(
                ["Current Quantity", "Total Inventory", "Total Retail", "Profit", "Profit Margin", "Velocity",
                 "Last 3 Months Sales", "Last 6 Months Sales"])

            # 特殊处理：Velocity 为0、无限大、空值或无效值用 '-' 替换
            def clean_velocity(x):
                if pd.isna(x) or x == 0 or x == float('inf') or x == float('-inf'):
                    return '-'
                return x

            df_low_display["Velocity"] = df_low_display["Velocity"].apply(clean_velocity)

            # === 保持float，0→NaN，显示'–'，不影响排序 ===
            for c in ["Total Inventory", "Total Retail", "Profit", "Velocity", "Last 3 Months Sales",
                      "Last 6 Months Sales"]:
                df_low_display[c] = pd.to_numeric(df_low_display[c], errors="coerce")
                df_low_display.loc[df_low_display[c].fillna(0) == 0, c] = pd.NA

            # ✅ Profit Margin 特殊处理
            if "Profit Margin" in df_low_display.columns:
                df_low_display["Profit Margin"] = (
                    df_low_display["Profit Margin"]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .replace("-", None)
                )
                df_low_display["Profit Margin"] = pd.to_numeric(df_low_display["Profit Margin"], errors="coerce")
                df_low_display.loc[df_low_display["Profit Margin"].fillna(0) == 0, "Profit Margin"] = pd.NA

            # 其他空值用字符 '-' 替换
            for col in display_columns:
                if col in df_low_display.columns:
                    if col not in ["Total Retail", "Total Inventory", "Profit", "Velocity",
                                   "Profit Margin", "Last 3 Months Sales", "Last 6 Months Sales"]:  # 这些列已经特殊处理过
                        df_low_display[col] = df_low_display[col].fillna('-')

            column_config = {
                'Item Name': st.column_config.Column(width=150),
                'Variation Name': st.column_config.Column(width=110),
                'Current Quantity': st.column_config.Column(width=110),
                'Total Inventory': st.column_config.NumberColumn("Total Inventory", width=100, format="%.1f"),
                'Total Retail': st.column_config.NumberColumn("Total Retail", width=80, format="%.1f"),
                'Profit': st.column_config.NumberColumn("Profit", width=50, format="%.1f"),
                'Profit Margin': st.column_config.NumberColumn("Profit Margin", width=90, format="%.1f%%"),
                'Velocity': st.column_config.NumberColumn("Velocity", width=60, format="%.1f"),
                'Last 3 Months Sales': st.column_config.NumberColumn("Last 3 Months Sales", width=120, format="$%.0f"),
                'Last 6 Months Sales': st.column_config.NumberColumn("Last 6 Months Sales", width=120, format="$%.0f"),
            }

            st.data_editor(
                df_low_display[display_columns],
                column_config=column_config,
                use_container_width=False,
                hide_index=True,
                disabled=True
            )
            
        else:
            st.info("No matching items found with the current filters.")
    else:
        st.success("No items found with the current filters.")

    st.markdown("---")

