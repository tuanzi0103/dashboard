import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import re
from typing import Union
from itertools import combinations

from services.db import get_db
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ==================== 可调参数 ====================
FORECAST_WEEKS = 3  # 预测3周
MIN_SERIES_LEN_FOR_HOLT = 10
RECENT_DAYS_FOR_VELOCITY = 30


# ==================== 小工具 ====================
def _persisting_multiselect(label, options, key):
    if key not in st.session_state:
        st.session_state[key] = []
    return st.multiselect(label, options=options, default=st.session_state[key], key=key)


def _item_col(df: pd.DataFrame) -> str:
    for c in ["Item", "Item Name", "Variation Name"]:
        if c in df.columns:
            return c
    return df.columns[0]


def _category_col(df: pd.DataFrame) -> Union[str, None]:
    for c in ["Category", "Categories", "Category Name", "Category (Top Level)",
              "Reporting Category", "Department"]:
        if c in df.columns:
            return c
    return None


def _order_key_cols(df: pd.DataFrame):
    for col in ["Order ID", "Receipt ID", "Txn ID", "Transaction ID"]:
        if col in df.columns:
            return col
    return None


def _format_dmy(x):
    if pd.isna(x) or x is None or str(x).strip() in ["", "0"]:
        return ""
    d = pd.to_datetime(x, errors="coerce")
    return "" if pd.isna(d) else d.strftime("%d/%m/%Y")


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


# ==================== 组合建议 ====================
def _build_category_pairs_by_order(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    df2 = df.copy()
    order_col = _order_key_cols(df2)
    if order_col is not None:
        order_key = df2[order_col].astype(str)
    else:
        if "Datetime" in df2.columns:
            key = pd.to_datetime(df2["Datetime"], errors="coerce").dt.floor("min").astype(str)
            if "Customer ID" in df2.columns:
                key = key + "_" + df2["Customer ID"].astype(str)
            order_key = key
        else:
            order_key = pd.Series(["__one__"] * len(df2), index=df2.index)
    df2 = df2[[cat_col]].assign(order_key=order_key)

    pair_counter = {}
    for _, g in df2.groupby("order_key"):
        cats = sorted(set(g[cat_col].dropna().astype(str).str.strip()))
        cats = [c for c in cats if c and c.lower() != "none"]
        if len(cats) < 2:
            continue
        for a, b in combinations(cats, 2):
            key = (a, b)
            pair_counter[key] = pair_counter.get(key, 0) + 1

    if not pair_counter:
        return pd.DataFrame(columns=["a", "b", "count"])

    return (pd.DataFrame([(k[0], k[1], v) for k, v in pair_counter.items()],
                         columns=["a", "b", "count"])
            .sort_values("count", ascending=False))


def _strategy_suggestions_by_category(df: pd.DataFrame, top_pairs: int = 6, slow_k: int = 6) -> dict:
    if df.empty:
        return {"popular_popular": [], "popular_slow": [], "discount_slow": []}

    cat_col = _category_col(df)
    if cat_col is None or "Qty" not in df.columns:
        return {"popular_popular": [], "popular_slow": [], "discount_slow": []}

    df = df[df[cat_col].notna()]
    df = df[df[cat_col].astype(str).str.strip().str.lower() != "none"]
    if df.empty:
        return {"popular_popular": [], "popular_slow": [], "discount_slow": []}

    cat_cnt = (df.groupby(cat_col)["Qty"]
               .sum()
               .reset_index(name="qty")
               .sort_values("qty", ascending=False))

    if cat_cnt.empty:
        return {"popular_popular": [], "popular_slow": [], "discount_slow": []}

    q75, q25 = cat_cnt["qty"].quantile(0.75), cat_cnt["qty"].quantile(0.25)
    popular = cat_cnt[cat_cnt["qty"] >= q75][cat_col].astype(str).tolist()
    slow = cat_cnt[cat_cnt["qty"] <= q25][cat_col].astype(str).tolist()  # 修复语法错误

    pairs = _build_category_pairs_by_order(df, cat_col)

    pp, ps = [], []
    if not pairs.empty and popular:
        mask = pairs["a"].isin(popular) & pairs["b"].isin(popular)
        pp = (pairs[mask].head(top_pairs)[["a", "b"]]
              .apply(lambda x: f"{x['a']} + {x['b']}", axis=1).tolist())

    if not pairs.empty and popular and slow:
        mask = ((pairs["a"].isin(popular) & pairs["b"].isin(slow)) |
                (pairs["a"].isin(slow) & pairs["b"].isin(popular)))
        ps = (pairs[mask].head(top_pairs)[["a", "b"]]
              .apply(lambda x: f"{x['a']} + {x['b']}", axis=1).tolist())

    if not ps and popular and slow:
        for a, b in zip(popular[:top_pairs], slow[:top_pairs]):
            ps.append(f"{a} + {b}")

    return {"popular_popular": pp, "popular_slow": ps, "discount_slow": slow[:slow_k]}


# ==================== 历史序列 & 预测 ====================
def _weekly_category_revenue(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    if "Net Sales" not in df.columns or "Datetime" not in df.columns:
        return pd.DataFrame(columns=["date", "value"])
    cat_col = _category_col(df)
    if cat_col is None:
        return pd.DataFrame(columns=["date", "value"])

    clean = [c.strip() for c in categories if str(c).strip() and str(c).strip().lower() != "none"]
    sub = df[df[cat_col].astype(str).isin(clean)]
    if sub.empty:
        return pd.DataFrame(columns=["date", "value"])
    sub = sub.copy()
    sub["date"] = pd.to_datetime(sub["Datetime"], errors="coerce").dt.to_period("W").dt.start_time
    return (sub.groupby("date")["Net Sales"].sum().reset_index(name="value")
            .sort_values("date"))


from sklearn.metrics import mean_absolute_percentage_error


def _holt_weekly_forecast(series_df: pd.DataFrame, forecast_weeks: int = FORECAST_WEEKS):
    if series_df.empty:
        return pd.DataFrame(columns=["date", "yhat"]), 0.0

    s = series_df.set_index(pd.to_datetime(series_df["date"]))["value"].asfreq("W").fillna(0.0)

    if s.size < MIN_SERIES_LEN_FOR_HOLT:
        # ⚡ 改为最近4周均线
        const = float(s.tail(4).mean()) if not s.empty else 0.0
        fut_idx = pd.date_range(start=(s.index.max() if len(s) else pd.Timestamp.today()) + pd.Timedelta(weeks=1),
                                periods=forecast_weeks, freq="W")
        return pd.DataFrame({"date": fut_idx.date, "yhat": [const] * forecast_weeks}), 0.0

    try:
        # ⚡ 用月度季节性（4 周一个周期）
        model = ExponentialSmoothing(s, trend="add", seasonal="add", seasonal_periods=4, damped_trend=True)
        fit = model.fit(optimized=True, use_brute=True)

        # ---- 计算预测准确率 ----
        split_point = int(len(s) * 0.8)
        train, test = s.iloc[:split_point], s.iloc[split_point:]
        if len(test) > 0:
            model_val = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=4,
                                             damped_trend=True).fit()
            pred = model_val.forecast(len(test))
            mape = 1 - mean_absolute_percentage_error(test, pred)
        else:
            mape = 0.0

        fut = fit.forecast(forecast_weeks)
        return pd.DataFrame({"date": fut.index.date, "yhat": fut.values}), float(round(mape, 3))

    except Exception:
        try:
            # fallback → Holt 线性趋势
            model = ExponentialSmoothing(s, trend="add")
            fit = model.fit()
            fut = fit.forecast(forecast_weeks)
            return pd.DataFrame({"date": fut.index.date, "yhat": fut.values}), 0.0
        except Exception:
            const = float(s.tail(4).mean()) if not s.empty else 0.0
            fut_idx = pd.date_range(start=s.index.max() + pd.Timedelta(weeks=1), periods=forecast_weeks, freq="W")
            return pd.DataFrame({"date": fut_idx.date, "yhat": [const] * forecast_weeks}), 0.0


# ==================== 缓存 ====================
@st.cache_data(show_spinner=False, persist=True)
def _precompute(tx):
    tx = tx.copy()
    tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")
    tx["date"] = tx["Datetime"].dt.date
    return tx


@st.cache_data(show_spinner=False, persist=True)
def load_inventory():
    conn = get_db()
    try:
        return pd.read_sql("SELECT * FROM inventory", conn)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, persist=True)
def compute_combo_forecast_category(tx, combo):
    cats = [s.strip() for s in str(combo).split("+") if s and s.strip()]
    series = _weekly_category_revenue(tx, cats)
    return _holt_weekly_forecast(series, forecast_weeks=FORECAST_WEEKS)


# ==================== 页面入口 ====================
def show_product_mix_only(tx: pd.DataFrame):
    # === 全局样式: 让 st.dataframe 里的所有表格文字左对齐 ===
    st.markdown("""
    <style>
    [data-testid="stDataFrame"] table {
        text-align: left !important;
    }
    [data-testid="stDataFrame"] th {
        text-align: left !important;
    }
    [data-testid="stDataFrame"] td {
        text-align: left !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style='font-size:22px; font-weight:700; margin-top:-2rem !important; margin-bottom:0.2rem !important;'>📊 Product Mix</h2>
    <style>
    /* 去掉 Streamlit 默认标题和上一个元素之间的间距 */
    div.block-container h2 {
        padding-top: 0 !important;
        margin-top: -2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stDataEditor"] table {
        table-layout: fixed !important;
    }
    [data-testid="stDataEditor"] td, 
    [data-testid="stDataEditor"] th {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if tx is None or tx.empty:
        st.info("No transaction data available.")
        return

    # 🔹 统一 Datetime/date 类型为 Timestamp
    tx["Datetime"] = pd.to_datetime(tx["Datetime"], errors="coerce")
    tx["date"] = tx["Datetime"].dt.normalize()  # 保持 Timestamp，不转成 date

    # --------- 建议 ---------
    st.markdown("<h3 style='font-size:18px; font-weight:700;'>💡 Discount Forecast Suggestions</h3>", unsafe_allow_html=True)
    cat_col = _category_col(tx)
    item_col = _item_col(tx)
    base_cols = [c for c in [cat_col or "", item_col, "Qty", "Datetime", "Customer ID"] if c and c in tx.columns]

    input_df = tx[base_cols].copy()
    if cat_col:
        mask_valid_cat = input_df[cat_col].notna() & (input_df[cat_col].astype(str).str.strip().str.lower() != "none")
        input_df = input_df[mask_valid_cat]

    sugg = _strategy_suggestions_by_category(input_df, top_pairs=6, slow_k=6)
    rows = []
    for p in sugg.get("popular_popular", []): rows.append({"strategy": "Bundle popular-popular", "combo": p})
    for p in sugg.get("popular_slow", []):    rows.append({"strategy": "Bundle popular-slow", "combo": p})
    for s in sugg.get("discount_slow", []):   rows.append({"strategy": "Discount slow mover", "combo": s})
    sugg_df = pd.DataFrame(rows, columns=["strategy", "combo"])
    if not sugg_df.empty:
        # === 设置 Discount Forecast Suggestions 表格列宽 ===
        suggestion_column_config = {
            "strategy": st.column_config.Column(width=160),
            "combo": st.column_config.Column(width=200)
        }
        st.data_editor(
            sugg_df,
            column_config=suggestion_column_config,
            use_container_width=False,
            hide_index=True,
            disabled=True
        )

    else:
        st.info("No suggestions available based on current data.")

    # --------- Inventory KPIs ---------
    st.markdown("<h3 style='font-size:18px; font-weight:700;'>📦 Inventory Details</h3>", unsafe_allow_html=True)

    conn = get_db()
    try:
        inv = pd.read_sql("SELECT * FROM inventory", conn)
    except Exception:
        inv = pd.DataFrame()

    if inv.empty:
        st.info("Inventory table not available.")
        return

    inv_key = "Item Name" if "Item Name" in inv.columns else None
    tx_key = "Item" if "Item" in tx.columns else None
    if not inv_key or not tx_key:
        st.info("Cannot align items between inventory and transactions.")
        return

    inv[inv_key] = _norm(inv[inv_key])
    tx[tx_key] = _norm(tx[tx_key])

    sold = tx.groupby(tx_key, as_index=False)["Qty"].sum().rename(columns={"Qty": "Sold"})
    last_sold = tx.dropna(subset=["Datetime"]).groupby(tx_key, as_index=False)["Datetime"].max().rename(
        columns={"Datetime": "Last sold"})

    df = inv.copy()
    out = pd.DataFrame({
        "Item Name": df[inv_key]
    })

    if "Current Quantity Vie Market & Bar" in inv.columns:
        onhand = pd.to_numeric(inv["Current Quantity Vie Market & Bar"], errors="coerce")
    elif "Stock on Hand" in inv.columns:
        onhand = pd.to_numeric(inv["Stock on Hand"], errors="coerce")
    else:
        onhand = np.nan
    out["On hand"] = onhand

    out = out.merge(sold, left_on="Item Name", right_on=tx_key, how="left")
    out = out.merge(last_sold, left_on="Item Name", right_on=tx_key, how="left")

    # === 修复售罄率计算 ===
    sold_num = pd.to_numeric(out.get("Sold", np.nan), errors="coerce").fillna(0)
    onh_num = pd.to_numeric(out.get("On hand", np.nan), errors="coerce").fillna(0)

    # ✅ 修复：负库存取绝对值
    onh_num_abs = onh_num.abs()

    # ✅ 修复：正确的售罄率计算公式
    # 售罄率 = 已售数量 / (已售数量 + 当前库存) * 100
    total_available = sold_num + onh_num_abs
    # 避免除零错误
    sell_through_ratio = np.where(total_available > 0, sold_num / total_available * 100, 0)

    # ✅ 修复：使用 pandas Series 进行字符串操作
    sell_through_rounded = pd.Series(sell_through_ratio.round(0).astype(int), index=out.index)
    out["Sell-through"] = sell_through_rounded.astype(str) + "%"

    # === 修复销售速度预测 ===
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(days=RECENT_DAYS_FOR_VELOCITY)

    # ✅ 修复：直接使用30天总销量作为月销量，避免逻辑循环
    monthly_sales_dict = (
        tx[tx["date"] >= cutoff]
        .groupby(tx_key)["Qty"].sum()
    ).to_dict()

    def calc_velocity(item):
        # ✅ 修复：直接返回30天总销量
        return int(monthly_sales_dict.get(item, 0))

    # === 修复缺货日期计算 ===
    def calc_out_of_stock(row):
        onhand = pd.to_numeric(row["On hand"], errors="coerce")
        monthly_sales = monthly_sales_dict.get(row["Item Name"], 0)

        # 计算日均销量（用于预测）
        daily_sales = monthly_sales / RECENT_DAYS_FOR_VELOCITY if monthly_sales > 0 else 0

        if pd.isna(onhand) or onhand <= 0:
            return today.strftime("%d/%m/%Y") + " (out of stock)"

        if daily_sales > 0:
            days_left = int(onhand / daily_sales)
            # 考虑安全库存（假设3天安全库存）
            safe_stock_days = 3
            if days_left <= safe_stock_days:
                return today.strftime("%d/%m/%Y") + " (need restock)"
            else:
                return (today + pd.Timedelta(days=days_left - safe_stock_days)).strftime("%d/%m/%Y")
        else:
            return "no sale records"

    # ✅ 修复：使用 pandas apply 方法
    out["Sales velocity"] = out["Item Name"].apply(calc_velocity).astype(str) + " per month"
    out["Out of stock"] = out.apply(calc_out_of_stock, axis=1)

    # 日期格式化
    for dcol in ["Last sold"]:
        if dcol in out.columns:
            out[dcol] = out[dcol].apply(_format_dmy)

    show_cols = ["Item Name", "Sell-through", "On hand", "Sold", "Sales velocity", "Last sold", "Out of stock"]

    # === 设置 Inventory Details 表格列宽 ===
    inventory_column_config = {
        "Item Name": st.column_config.Column(width=250),
        "Sell-through": st.column_config.Column(width=90),
        "On hand": st.column_config.Column(width=60),
        "Sold": st.column_config.Column(width=40),
        "Sales velocity": st.column_config.Column(width=100),
        "Last sold": st.column_config.Column(width=80),
        "Out of stock": st.column_config.Column(width=100)
    }

    st.data_editor(
        out[show_cols],
        column_config=inventory_column_config,
        use_container_width=False,
        hide_index=True,
        disabled=True
    )
