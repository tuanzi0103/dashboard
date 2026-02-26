import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st

# 保留原有的函数 simulate_transactions / simulate_consumption / simulate_consumption_timeseries 不动

@st.cache_data(show_spinner=False)
def simulate_combo_revenue(combos: list[str], months: int = 1) -> pd.DataFrame:
    """
    生成随机模拟的组合收入曲线数据
    - combos: 商品组合列表（["A + B", "C + D", ...]）
    - months: 生成多少个月的数据 (1,3,6,9)
    返回: DataFrame [date, combo, revenue]
    """
    days = months * 30
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=days-1)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    rng = np.random.default_rng(seed=2025)
    records = []

    for combo in combos:
        base = rng.integers(300, 800)  # 每个组合一个基准值
        for d in dates:
            rev = rng.normal(loc=base, scale=base * 0.2)  # 加20%波动
            records.append([d, combo, max(0, rev)])

    df = pd.DataFrame(records, columns=["date", "combo", "revenue"])
    return df


@st.cache_data(show_spinner=False)
def simulate_transactions(tx: pd.DataFrame, months: int = 1) -> pd.DataFrame:
    """原有：保留"""
    days = months * 30
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=days-1)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    if tx is None or tx.empty:
        base_sales = 1000
    else:
        base_sales = tx["Net Sales"].mean() if "Net Sales" in tx.columns else 1000

    rng = np.random.default_rng(seed=42)
    sales = rng.normal(loc=base_sales, scale=base_sales * 0.2, size=len(dates)).clip(min=0)
    qty = rng.poisson(lam=50, size=len(dates))
    cats = rng.choice(["retail", "bar", "other"], size=len(dates))

    df = pd.DataFrame({
        "Datetime": dates,
        "Net Sales": sales,
        "Gross Sales": sales * 1.2,
        "Discounts": -sales * 0.05,
        "Qty": qty,
        "Category": cats,
        "location": cats,
    })
    return df


@st.cache_data(show_spinner=False)
def simulate_consumption(inventory: pd.DataFrame, months: int = 1) -> pd.DataFrame:
    """原有：保留"""
    days = months * 30
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=days-1)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    rng = np.random.default_rng(seed=123)
    qty = rng.poisson(lam=40, size=len(dates))

    if inventory is not None and not inventory.empty:
        item_col = "Item" if "Item" in inventory.columns else "Item Name"
        item_pool = inventory[item_col].dropna().astype(str).unique()
    else:
        item_pool = ["SimItemA", "SimItemB"]

    df = pd.DataFrame({
        "Datetime": dates,
        "Qty": qty,
        "Item": rng.choice(item_pool, size=len(dates)),
        "SKU": rng.integers(1000, 2000, size=len(dates)).astype(str),
    })
    return df


def simulate_consumption_timeseries(inventory: pd.DataFrame, months: int = 1, items=None):
    """原有：保留"""
    df = simulate_consumption(inventory, months=months)
    df["date"] = df["Datetime"].dt.floor("D")
    df = df.groupby(["date", "Item"], as_index=False)["Qty"].sum().rename(columns={"Qty": "qty"})
    df["qty"] = df["qty"].abs()

    if items:
        df = df[df["Item"].isin(items)]

    if df.empty:
        rng = np.random.default_rng(seed=111)
        dates = pd.date_range(start=pd.Timestamp.today().normalize() - pd.Timedelta(days=30),
                              periods=30, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "Item": ["SimFallback"] * len(dates),
            "qty": rng.poisson(lam=30, size=len(dates))
        })

    rng = np.random.default_rng(seed=456)
    ds = df.copy()
    ds["qty"] = ds["qty"] + rng.normal(0, 5, size=len(ds))

    fc_list = []
    for item, grp in ds.groupby("Item"):
        grp = grp.sort_values("date")
        last_week = grp.tail(7)
        avg_qty = last_week["qty"].mean() if not last_week.empty else grp["qty"].mean()
        base = abs(avg_qty)

        future_dates = pd.date_range(start=grp["date"].max() + pd.Timedelta(days=1), periods=30)
        rng2 = np.random.default_rng(seed=789)
        noise = rng2.normal(0, base * 0.2, size=30)
        forecast = pd.DataFrame({
            "date": future_dates,
            "forecast_qty": base + noise,
            "Item": item
        })
        fc_list.append(forecast)

    fc = pd.concat(fc_list, ignore_index=True) if fc_list else pd.DataFrame(columns=["date", "forecast_qty", "Item"])
    return ds, fc


# ==================== 🔹 新增：策略收入曲线模拟 ====================
@st.cache_data(show_spinner=False)
def simulate_strategy_revenue_curves(tx: pd.DataFrame, strategies: dict, days: int = 30) -> pd.DataFrame:
    """
    根据历史销售（Net Sales）得到 baseline 的未来 30 天曲线，
    并基于三类策略（popular-popular / popular-slow / discount slow mover）的“组合数量”
    叠加预期增益，返回长表：
    columns: [date, strategy, revenue]，strategy 取
        - baseline
        - bundle_popular_popular
        - bundle_popular_slow
        - discount_slow_mover
    """
    # 1) baseline：从历史日销售推断（Holt-Winters，有则用；否则均值）
    df = tx.copy()
    if df.empty or "Datetime" not in df.columns or "Net Sales" not in df.columns:
        # 兜底：构造常数基线
        date_index = pd.date_range(end=pd.Timestamp.today().normalize() + pd.Timedelta(days=days),
                                   periods=days, freq="D")
        base = pd.Series(np.full(days, 1000.0), index=date_index)
    else:
        df["date"] = df["Datetime"].dt.floor("D")
        daily = df.groupby("date", as_index=False)["Net Sales"].sum().sort_values("date")
        y = daily.set_index("date")["Net Sales"].tail(180)
        if len(y) >= 14:
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=7)
                fit = model.fit()
                base = fit.forecast(days)
            except Exception:
                base = pd.Series([y.mean()] * days,
                                 index=pd.date_range(start=y.index.max() + pd.Timedelta(days=1), periods=days))
        else:
            base = pd.Series([y.mean()] * days,
                             index=pd.date_range(start=y.index.max() + pd.Timedelta(days=1), periods=days))

    # 2) 根据策略组合数量设置增益（带上限），并加入轻微随机扰动，避免死板直线
    n_pp = len(strategies.get("popular_popular", []) or [])
    n_ps = len(strategies.get("popular_slow", []) or [])
    n_ds = len(strategies.get("discount_slow", []) or [])

    uplift_pp = min(0.04 * n_pp, 0.20)     # 热销×热销：每个组合+4%，最多+20%
    uplift_ps = min(0.025 * n_ps, 0.125)   # 热销×慢销：每个组合+2.5%，最多+12.5%
    uplift_ds = min(0.015 * max(n_ds, 1), 0.10)  # 慢销降价：每个单品+1.5%，最多+10%

    idx = base.index
    rng = np.random.default_rng(seed=20250928)
    jitter = lambda scale: 1.0 + rng.normal(0, scale, size=len(idx))  # 轻微上下浮动

    series = {
        "baseline": (base * jitter(0.02)).clip(lower=0),
        "bundle_popular_popular": (base * (1 + uplift_pp) * jitter(0.025)).clip(lower=0),
        "bundle_popular_slow": (base * (1 + uplift_ps) * jitter(0.025)).clip(lower=0),
        "discount_slow_mover": (base * (1 + uplift_ds) * jitter(0.03)).clip(lower=0),
    }

    # 3) 转成长表
    out = []
    for name, s in series.items():
        out.append(pd.DataFrame({"date": s.index, "strategy": name, "revenue": s.values}))
    fut_long = pd.concat(out, ignore_index=True)
    return fut_long
