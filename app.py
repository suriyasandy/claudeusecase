"""
Accounting Control AI Platform — Phase 1
=========================================
Built on the uploaded Phase_1_app.txt with two major additions:

NEW FEATURE 1: CLICK-THROUGH DRILL-DOWN (Tab 3 — Break Counts)
  - Top chart shows all top-10 Rec Names trend (unchanged)
  - Below it: user selects ANY Rec Name from a dropdown
  - Dashboard renders full granular breakdown of that Rec:
    counts/amounts by Period × Team, Period × Entity,
    Period × Type of Break, Period × Asset Class
  - Uses st.session_state to persist selection across reruns

NEW FEATURE 2: STATISTICAL FALSE POSITIVE THRESHOLDING (Tab 6 — new)
  - For each segment (Rec + Team + Entity + Asset Class):
      * Computes break count across all uploaded periods
      * Uses last N-1 periods as "historical baseline"
      * Flags latest period as FP CANDIDATE if count is within
        historical_mean ± k × MAD  (configurable k)
      * Three confidence tiers: High / Medium / Low FP confidence
  - User reviews the ranked FP candidates and confirms/rejects
    via data_editor checkboxes
  - Confirmed FPs propagate to sidebar "Exclude False Positives" filter

NEW FEATURE 3: PERIOD COMPARISON (Tab 7 — new)
  - Loads cached historical period data
  - Compares latest upload vs historical average
  - Waterfall delta charts by Rec Name and Team

ALL EXISTING FIXES RETAINED:
  - from __future__ import annotations  (Python 3.9 compatible)
  - format_short() K/M/B/T/P  (no 10^16 on axes)
  - TRY_CAST AS DOUBLE in all DuckDB amount SQL
  - Apply-button filter pattern
  - Auto Parquet conversion and cache
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import date as _date
from io import BytesIO

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
    HAS_AGGRID = True
except ImportError:
    HAS_AGGRID = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Accounting Control AI Platform",
    page_icon="📊", layout="wide",
    initial_sidebar_state="expanded",
)

COLORS       = ["#01696F","#0C4E54","#20808D","#4F98A3","#A84B2F",
                "#1B474D","#BCE2E7","#944454","#FFC553","#5C2D91"]
PRIMARY      = "#01696F"
DARK         = "#28251D"
MUTED        = "#7A7974"
BG_LIGHT     = "#F7F7F5"
WARN         = "#A84B2F"
BUCKET_ORDER = ["0-15","16-30","31-60","61-90","91-180","181-365","365+","Unknown"]
MAX_JS_INT   = 2 ** 53

st.markdown(f"""
<style>
  .main .block-container{{padding-top:1rem;padding-bottom:1rem;}}
  h1{{color:{DARK};font-size:1.6rem !important;}}
  h2{{color:{DARK};font-size:1.25rem !important;}}
  h3{{color:{DARK};font-size:1.05rem !important;}}
  .stTabs [data-baseweb="tab-list"]{{gap:8px;}}
  .stTabs [data-baseweb="tab"]{{background-color:{BG_LIGHT};border-radius:6px 6px 0 0;
      padding:8px 16px;font-weight:600;color:{DARK};}}
  .stTabs [aria-selected="true"]{{background-color:{PRIMARY} !important;color:white !important;}}
  .kpi-card{{background:white;border-left:4px solid {PRIMARY};padding:12px 16px;
      border-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.08);}}
  .kpi-warn{{background:white;border-left:4px solid {WARN};padding:12px 16px;
      border-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.08);}}
  .kpi-label{{color:{MUTED};font-size:.75rem;text-transform:uppercase;
      letter-spacing:.5px;margin-bottom:2px;}}
  .kpi-value{{color:{DARK};font-size:1.4rem;font-weight:700;margin:0;}}
  .kpi-delta-pos{{color:#A84B2F;font-size:.8rem;}}
  .kpi-delta-neg{{color:#01696F;font-size:.8rem;}}
  .banner-warn{{background:#FDECEA;border-left:4px solid {WARN};padding:10px 16px;
      border-radius:4px;margin-bottom:.5rem;font-size:.88rem;color:{DARK};}}
  .banner-info{{background:#E6F4F1;border-left:4px solid {PRIMARY};padding:10px 16px;
      border-radius:4px;margin-bottom:.5rem;font-size:.88rem;color:{DARK};}}
  .banner-fp{{background:#FFF8E1;border-left:4px solid #FFC553;padding:10px 16px;
      border-radius:4px;margin-bottom:.5rem;font-size:.88rem;color:{DARK};}}
  .banner-drill{{background:#EDE7F6;border-left:4px solid #5C2D91;padding:10px 16px;
      border-radius:4px;margin-bottom:.5rem;font-size:.88rem;color:{DARK};}}
  .banner-jira{{background:#E8F4FD;border-left:4px solid #20808D;padding:10px 16px;
      border-radius:4px;margin-bottom:.5rem;font-size:.88rem;color:{DARK};}}
  div[data-testid="stButton"]>button{{background-color:{PRIMARY};color:white;
      border:none;font-weight:600;border-radius:6px;padding:8px 24px;}}
  div[data-testid="stButton"]>button:hover{{background-color:#0C4E54;color:white;}}
  .drill-section{{background:#F8F6FF;border:1px solid #D1C4E9;border-radius:8px;
      padding:16px;margin-top:12px;}}
</style>
""", unsafe_allow_html=True)


# ── Utility functions ─────────────────────────────────────────────────────────

def format_short(val) -> str:
    """Format large numbers with K/M/B/T/P suffixes."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if math.isnan(v) or math.isinf(v):
        return "N/A"
    abs_v = abs(v)
    if abs_v >= 1e15:
        return f"{v/1e15:.1f}P"
    if abs_v >= 1e12:
        return f"{v/1e12:.1f}T"
    if abs_v >= 1e9:
        return f"{v/1e9:.1f}B"
    if abs_v >= 1e6:
        return f"{v/1e6:.1f}M"
    if abs_v >= 1e3:
        return f"{v/1e3:.1f}K"
    return f"{v:,.0f}"


def format_number(val, decimals: int = 0) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if math.isnan(v) or math.isinf(v):
        return "N/A"
    return f"{v:,.{decimals}f}"


def safe_mom_pct(curr, prev):
    """Return MoM % change or None if prev is zero/None."""
    try:
        c, p = float(curr), float(prev)
        if p == 0:
            return None
        return round((c - p) / abs(p) * 100, 1)
    except Exception:
        return None


def kpi_card(label: str, value: str, delta_str: str = "", invert: bool = False, warn: bool = False) -> None:
    card_cls = "kpi-warn" if warn else "kpi-card"
    delta_html = ""
    if delta_str:
        delta_html = f'<div class="kpi-label" style="margin-top:4px;">{delta_str}</div>'
    st.markdown(
        f'<div class="{card_cls}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{delta_html}</div>',
        unsafe_allow_html=True,
    )


def chart_layout(fig, title: str, xlab: str = "", ylab: str = "", height: int = 400):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=DARK)),
        xaxis_title=xlab, yaxis_title=ylab,
        height=height,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color=DARK, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E0E0E0")
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0", linecolor="#E0E0E0")
    return fig


def add_rolling_avg(fig, df, x_col: str, y_col: str, name: str = "Rolling Avg"):
    if len(df) >= 3:
        roll = df[y_col].rolling(3, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df[x_col], y=roll, mode="lines", name=name,
            line=dict(color=WARN, width=2, dash="dot"),
        ))
    return fig


def render_grid(df: pd.DataFrame, height: int = 400, key: str = "grid") -> None:
    if HAS_AGGRID:
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(resizable=True, sortable=True, filter=True)
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
        AgGrid(df, gridOptions=gb.build(), height=height, key=key,
               allow_unsafe_jscode=True, theme="streamlit")
    else:
        st.dataframe(df, use_container_width=True, height=height)


# ── DuckDB setup ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_duck():
    return duckdb.connect()


_last_tbl_id = -1


def dq(sql, df=None):
    """Run SQL against the shared DuckDB connection with id-based re-registration guard."""
    global _last_tbl_id
    con = get_duck()
    if df is not None and id(df) != _last_tbl_id:
        con.register("tbl", df)
        _last_tbl_id = id(df)
    return con.execute(sql).df()


def dq_local(sql: str, **named_dfs) -> pd.DataFrame:
    """Execute SQL with named DataFrames in an isolated DuckDB connection (thread-safe)."""
    con = duckdb.connect()
    for name, frame in named_dfs.items():
        con.register(name, frame)
    return con.execute(sql).df()


def safe_amt(col: str) -> str:
    return f"TRY_CAST(\"{col}\" AS DOUBLE)"


# ── Parquet cache functions ───────────────────────────────────────────────────

def _cache_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), "acct_ai_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _pq_path(fhash: str) -> str:
    return os.path.join(_cache_dir(), f"{fhash}.parquet")


def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _period_index_path():
    return os.path.join(_cache_dir(), "period_index.json")


def update_period_index(fhash: str, period_labels: list) -> None:
    """Record which periods are in each cached Parquet file."""
    path = _period_index_path()
    try:
        idx = json.loads(open(path).read()) if os.path.exists(path) else {}
    except Exception:
        idx = {}
    for p in period_labels:
        idx[p] = fhash
    try:
        with open(path, "w") as f:
            json.dump(idx, f)
    except Exception:
        pass


def load_historical_context(current_periods: list, current_fhash: str):
    """Load cached period data NOT in the current upload for period comparison."""
    path = _period_index_path()
    if not os.path.exists(path):
        return None
    try:
        idx = json.loads(open(path).read())
    except Exception:
        return None
    hist_hashes = {h for p, h in idx.items()
                   if p not in set(current_periods) and h != current_fhash}
    frames = []
    for h in hist_hashes:
        pq = _pq_path(h)
        if os.path.exists(pq):
            try:
                frames.append(pd.read_parquet(pq))
            except Exception:
                pass
    if not frames:
        return None
    hist_df = pd.concat(frames, ignore_index=True)
    # Restore categorical bucket
    if "_Computed_Bucket" in hist_df.columns:
        hist_df["_Computed_Bucket"] = pd.Categorical(
            hist_df["_Computed_Bucket"], categories=BUCKET_ORDER, ordered=True)
    return hist_df


# ── Vectorised helpers ────────────────────────────────────────────────────────

def period_to_datetime_vec(series):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.notna() & s.between(190001, 209912)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if valid.any():
        yyyymm = s[valid].astype(int).astype(str).str.zfill(6)
        result[valid] = (
            pd.to_datetime(yyyymm, format="%Y%m", errors="coerce")
            + pd.offsets.MonthEnd(0)
        )
    return result


def parse_amount_vec(series) -> tuple[pd.Series, int]:
    """Parse amount column to float. Returns (series, overflow_count)."""
    r = pd.to_numeric(series, errors="coerce")
    overflow = r.abs() > MAX_JS_INT
    overflow_count = int(overflow.sum())
    return r, overflow_count


def parse_age_days_vec(series) -> pd.Series:
    r = pd.to_numeric(series, errors="coerce")
    r = r.where(r.between(0, 36500), other=np.nan)
    return r


def age_to_bucket_vec(age_days: pd.Series) -> pd.Categorical:
    bins   = [-1, 15, 30, 60, 90, 180, 365, float("inf")]
    labels = ["0-15","16-30","31-60","61-90","91-180","181-365","365+"]
    bucketed = pd.cut(age_days, bins=bins, labels=labels)
    result = bucketed.astype(object).fillna("Unknown")
    return pd.Categorical(result, categories=BUCKET_ORDER, ordered=True)


# ── Column mapping ────────────────────────────────────────────────────────────

EXPECTED_COLUMNS = {
    "Rec Name (as per Rec Cube)": ["rec name", "rec_name", "reconciliation name", "rec cube"],
    "Team": ["team", "team name", "group"],
    "Entity": ["entity", "legal entity", "entity name"],
    "Type of Break": ["type of break", "break type", "break_type"],
    "Asset Class": ["asset class", "asset_class", "assetclass"],
    "Date": ["date", "break date", "trade date", "value date"],
    "Period": ["period", "reporting period", "month"],
    "Age Days": ["age days", "age_days", "days aged", "days old"],
    "ABS GBP": ["abs gbp", "abs_gbp", "absolute gbp"],
    "BREAK AMOUNT GBP": ["break amount gbp", "break_amount_gbp", "amount gbp", "gbp amount"],
    "Jira Flag": ["jira flag", "jira_flag", "has jira", "jira"],
    "Jira ID": ["jira id", "jira_id", "ticket id", "ticket"],
    "Comments": ["comments", "comment", "notes", "note"],
    "_FP_Confirmed": ["_fp_confirmed", "fp confirmed", "false positive confirmed"],
}


def fuzzy_match(col_name: str, candidates: list[str]) -> bool:
    cn = col_name.lower().strip()
    return any(c in cn or cn in c for c in candidates)


def build_col_map(df: pd.DataFrame) -> dict[str, str]:
    col_map = {}
    for expected, variants in EXPECTED_COLUMNS.items():
        for actual in df.columns:
            if fuzzy_match(actual, variants):
                col_map[expected] = actual
                break
    return col_map


def drop_blank_trailing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where all non-index columns are NaN."""
    return df.dropna(how="all")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(raw_bytes: bytes) -> dict:
    fhash = file_hash(raw_bytes)
    pq = _pq_path(fhash)

    if os.path.exists(pq):
        df = pd.read_parquet(pq)
        if "_Computed_Bucket" in df.columns:
            df["_Computed_Bucket"] = pd.Categorical(
                df["_Computed_Bucket"], categories=BUCKET_ORDER, ordered=True)
        col_map = build_col_map(df)
        return {"df": df, "col_map": col_map, "fhash": fhash, "overflow_count": 0, "cached": True}

    # Parse raw bytes
    try:
        df = pd.read_excel(BytesIO(raw_bytes), engine="openpyxl")
    except Exception:
        try:
            df = pd.read_csv(BytesIO(raw_bytes))
        except Exception as e:
            st.error(f"Cannot parse file: {e}")
            st.stop()

    df = drop_blank_trailing(df)
    col_map = build_col_map(df)

    total_overflow = 0

    # Parse Period
    period_actual = col_map.get("Period")
    if period_actual and period_actual in df.columns:
        df["_Period_dt"] = period_to_datetime_vec(df[period_actual])
        df["_Period_label"] = df["_Period_dt"].dt.strftime("%Y-%m").fillna("Unknown")
    else:
        df["_Period_dt"] = pd.NaT
        df["_Period_label"] = "Unknown"

    # Parse Date
    date_actual = col_map.get("Date")
    if date_actual and date_actual in df.columns:
        df[date_actual] = pd.to_datetime(df[date_actual], errors="coerce")

    # Parse amounts
    for key in ["ABS GBP", "BREAK AMOUNT GBP"]:
        actual = col_map.get(key)
        if actual and actual in df.columns:
            df[actual], ov = parse_amount_vec(df[actual])
            total_overflow += ov

    # Parse age days
    age_actual = col_map.get("Age Days")
    if age_actual and age_actual in df.columns:
        df[age_actual] = parse_age_days_vec(df[age_actual])
        df["_Computed_Age_Days"] = df[age_actual]
    elif date_actual and date_actual in df.columns:
        today = pd.Timestamp.today()
        df["_Computed_Age_Days"] = (today - df[date_actual]).dt.days

    if "_Computed_Age_Days" in df.columns:
        df["_Computed_Bucket"] = age_to_bucket_vec(df["_Computed_Age_Days"])
    else:
        df["_Computed_Bucket"] = pd.Categorical(
            ["Unknown"] * len(df), categories=BUCKET_ORDER, ordered=True)

    # FP column
    fp_actual = col_map.get("_FP_Confirmed")
    if fp_actual and fp_actual in df.columns:
        df["_FP_Confirmed"] = df[fp_actual].astype(bool)
    else:
        df["_FP_Confirmed"] = False

    # Update period index for incremental comparison
    if "_Period_label" in df.columns:
        current_periods = df["_Period_label"].dropna().unique().tolist()
        update_period_index(fhash, current_periods)

    # Save parquet
    try:
        df.to_parquet(pq, index=False)
    except Exception:
        pass

    return {"df": df, "col_map": col_map, "fhash": fhash, "overflow_count": total_overflow, "cached": False}


# ── Filter pattern ────────────────────────────────────────────────────────────

def build_sidebar_filters(df: pd.DataFrame, col_map: dict) -> dict:
    st.sidebar.markdown("## Filters")

    filters = {}

    # Period filter
    if "_Period_label" in df.columns:
        periods = sorted(df["_Period_label"].dropna().unique().tolist())
        if periods:
            sel = st.sidebar.multiselect("Period", periods, default=periods, key="_f_period")
            if sel:
                filters["_Period_label"] = sel

    # Categorical filters
    for key in ["Team", "Entity", "Rec Name (as per Rec Cube)", "Type of Break", "Asset Class"]:
        actual = col_map.get(key)
        if actual and actual in df.columns:
            opts = sorted(df[actual].dropna().astype(str).unique().tolist())
            if opts:
                sel = st.sidebar.multiselect(key, opts, default=opts, key=f"_f_{key}")
                if sel and len(sel) < len(opts):
                    filters[actual] = sel

    # False positive exclusion
    st.sidebar.markdown("---")
    excl_fp = st.sidebar.checkbox("Exclude Confirmed False Positives", value=False, key="_excl_fp")
    if excl_fp:
        filters["_EXCL_FP"] = True

    st.sidebar.markdown("---")
    if st.sidebar.button("Apply Filters", key="_apply_btn"):
        st.session_state["_filters_applied"] = filters
        st.rerun()

    return st.session_state.get("_filters_applied", {})


def _reset_filters():
    for k in ["_filters_applied", "_fp_seg_keys", "_fp_seg_cols",
              "_drill_rec", "_overflow"]:
        if k in st.session_state:
            del st.session_state[k]


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    where = []
    # Build filter using pandas directly for simplicity and thread safety
    result = df.copy()
    for key, val in filters.items():
        if key == "_EXCL_FP":
            fp_keys = st.session_state.get("_fp_seg_keys", [])
            fp_cols = st.session_state.get("_fp_seg_cols", [])
            if fp_keys and fp_cols:
                excl_mask = pd.Series(True, index=result.index)
                for seg_key in fp_keys:
                    seg_mask = pd.Series(True, index=result.index)
                    for c, v in zip(fp_cols, seg_key):
                        if c in result.columns:
                            seg_mask &= result[c].astype(str) == str(v)
                    excl_mask &= ~seg_mask
                result = result[excl_mask]
            else:
                if "_FP_Confirmed" in result.columns:
                    result = result[~result["_FP_Confirmed"]]
        elif key in result.columns:
            result = result[result[key].astype(str).isin([str(v) for v in val])]
    return result


# ── Tab: Data Quality ─────────────────────────────────────────────────────────

def tab_data_quality(df: pd.DataFrame, df_f: pd.DataFrame, col_map: dict) -> None:
    st.markdown("## 🧹 Data Quality Report")

    total_rows = len(df)
    filtered_rows = len(df_f)
    n_cols = len(df.columns)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Total Rows (raw)", format_number(total_rows))
    with k2:
        kpi_card("Filtered Rows", format_number(filtered_rows))
    with k3:
        kpi_card("Columns", format_number(n_cols))
    with k4:
        mapped = len([k for k in EXPECTED_COLUMNS if k in col_map])
        kpi_card("Mapped Columns", f"{mapped} / {len(EXPECTED_COLUMNS)}")

    st.markdown("---")
    st.markdown("### Column Mapping")
    mapping_rows = []
    for exp, variants in EXPECTED_COLUMNS.items():
        actual = col_map.get(exp)
        mapping_rows.append({
            "Expected Column": exp,
            "Mapped To": actual or "❌ Not Found",
            "Status": "✅ Mapped" if actual else "⚠️ Missing",
        })
    st.dataframe(pd.DataFrame(mapping_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Null / Missing Value Summary")
    null_rows = []
    for col in df_f.columns:
        if col.startswith("_"):
            continue
        n_null = df_f[col].isna().sum()
        pct = n_null / max(len(df_f), 1) * 100
        null_rows.append({"Column": col, "Null Count": n_null, "Null %": round(pct, 1)})
    null_df = pd.DataFrame(null_rows).sort_values("Null %", ascending=False)
    st.dataframe(null_df, use_container_width=True, hide_index=True)

    overflow_count = st.session_state.get("_overflow", 0)
    if overflow_count and overflow_count > 0:
        st.markdown(
            f'<div class="banner-warn">⚠️ {overflow_count} amount value(s) exceed JavaScript safe integer '
            f'({MAX_JS_INT:,}) and may display imprecisely in charts.</div>',
            unsafe_allow_html=True)


# ── Tab: Ageing Validation ────────────────────────────────────────────────────

def tab_ageing_validation(df_f: pd.DataFrame, col_map: dict) -> None:
    st.markdown("## 📅 Ageing Validation")

    if "_Computed_Bucket" not in df_f.columns:
        st.info("No age data available. Ensure 'Age Days' or 'Date' column is present.")
        return

    # Custom As-Of Date (Change 11)
    st.markdown("### ⚙️ Ageing Configuration")
    as_of_col, _ = st.columns([2, 4])
    with as_of_col:
        as_of_date = st.date_input(
            "Compute Age As-Of Date",
            value=_date.today(),
            key="as_of_date",
            help="Age Days = selected date − original break Date. Defaults to today."
        )
    as_of_ts = pd.Timestamp(as_of_date)
    date_actual = col_map.get("Date")
    if date_actual and date_actual in df_f.columns and pd.api.types.is_datetime64_any_dtype(df_f[date_actual]):
        df_f = df_f.copy()
        df_f["_Computed_Age_Days"] = (as_of_ts - df_f[date_actual]).dt.days
        df_f["_Computed_Bucket"] = age_to_bucket_vec(df_f["_Computed_Age_Days"])
        if as_of_date != _date.today():
            st.markdown(
                f'<div class="banner-info">📅 Age computed as of <b>{as_of_date}</b>. '
                f'All ageing KPIs and charts below reflect this date.</div>',
                unsafe_allow_html=True)

    bucket_counts = (
        df_f.groupby("_Computed_Bucket", observed=True)
        .size()
        .reindex(BUCKET_ORDER, fill_value=0)
        .reset_index()
    )
    bucket_counts.columns = ["Bucket", "Count"]

    # KPIs
    total = len(df_f)
    old_mask = df_f["_Computed_Bucket"].isin(["91-180","181-365","365+"])
    old_count = int(old_mask.sum())
    old_pct = old_count / max(total, 1) * 100

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("Total Breaks", format_number(total))
    with k2:
        kpi_card("Breaks >90 Days", format_number(old_count),
                 f"{old_pct:.1f}% of total", warn=old_pct > 20)
    with k3:
        if "_Computed_Age_Days" in df_f.columns:
            avg_age = df_f["_Computed_Age_Days"].mean()
            kpi_card("Avg Age (Days)", format_number(avg_age, 1))

    st.markdown("---")

    fig = px.bar(
        bucket_counts, x="Bucket", y="Count",
        color_discrete_sequence=[PRIMARY],
        text="Count",
    )
    fig = chart_layout(fig, "Break Count by Age Bucket", "Age Bucket", "Count", height=380)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # By Team
    team_actual = col_map.get("Team")
    if team_actual and team_actual in df_f.columns:
        st.markdown("### Age Distribution by Team")
        team_bucket = (
            df_f.groupby([team_actual, "_Computed_Bucket"], observed=True)
            .size()
            .reset_index(name="Count")
        )
        fig2 = px.bar(
            team_bucket, x=team_actual, y="Count", color="_Computed_Bucket",
            color_discrete_sequence=COLORS,
        )
        fig2 = chart_layout(fig2, "Age Bucket by Team", team_actual, "Count", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Period trend of ageing
    if "_Period_label" in df_f.columns:
        st.markdown("### Avg Age Days by Period")
        age_trend = (
            df_f.groupby("_Period_label")["_Computed_Age_Days"]
            .mean()
            .reset_index()
            .sort_values("_Period_label")
        )
        age_trend.columns = ["Period", "Avg Age Days"]
        fig3 = px.line(age_trend, x="Period", y="Avg Age Days",
                       markers=True, color_discrete_sequence=[PRIMARY])
        fig3 = chart_layout(fig3, "Average Age Days Trend by Period", "Period", "Avg Age Days", height=340)
        st.plotly_chart(fig3, use_container_width=True)


# ── Tab: Break Counts + Drill-Down ────────────────────────────────────────────

def tab_break_counts(df_f: pd.DataFrame, col_map: dict) -> None:
    st.markdown("## 📈 Break Counts + Drill-Down")

    rec_actual  = col_map.get("Rec Name (as per Rec Cube)")
    team_actual = col_map.get("Team")

    if "_Period_label" not in df_f.columns:
        st.info("No period data available.")
        return

    # Top-10 Rec Names trend
    if rec_actual and rec_actual in df_f.columns:
        top10_series = df_f[rec_actual].value_counts().head(10).index.tolist()
        top10_df = df_f[df_f[rec_actual].isin(top10_series)]
        trend = (
            top10_df.groupby(["_Period_label", rec_actual])
            .size()
            .reset_index(name="Count")
            .sort_values("_Period_label")
        )
        fig = px.line(
            trend, x="_Period_label", y="Count", color=rec_actual,
            color_discrete_sequence=COLORS, markers=True,
        )
        fig = chart_layout(fig, "Top-10 Rec Names — Break Count Trend", "Period", "Count", height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        # ── Drill-Down Section ────────────────────────────────────────────────
        st.markdown("### 🔍 Drill-Down by Rec Name")
        all_recs = sorted(df_f[rec_actual].dropna().astype(str).unique().tolist())

        # Validate session state before rendering selectbox (Change 10)
        if st.session_state.get("_drill_rec") not in all_recs:
            st.session_state["_drill_rec"] = top10_series[0] if top10_series else (all_recs[0] if all_recs else None)

        if all_recs:
            selected_rec = st.selectbox(
                "Select Rec Name to drill into:",
                all_recs,
                index=all_recs.index(st.session_state["_drill_rec"]) if st.session_state.get("_drill_rec") in all_recs else 0,
                key="_drill_rec",
            )

            st.markdown(
                f'<div class="banner-drill">🔬 Drilling into: <b>{selected_rec}</b></div>',
                unsafe_allow_html=True)

            rec_df = df_f[df_f[rec_actual].astype(str) == selected_rec].copy()

            if len(rec_df) == 0:
                st.warning("No data for selected Rec Name.")
            else:
                st.markdown(f"**{format_number(len(rec_df))} breaks** in this Rec")

                with st.container():
                    st.markdown('<div class="drill-section">', unsafe_allow_html=True)

                    # 1. Period × Team
                    if team_actual and team_actual in rec_df.columns:
                        period_cnt = dq_local(
                            f'SELECT "_Period_label", "{team_actual}", COUNT(*) AS cnt '
                            f'FROM rec_tbl GROUP BY "_Period_label", "{team_actual}" '
                            f'ORDER BY "_Period_label"',
                            rec_tbl=rec_df
                        )
                        fig_pt = px.bar(
                            period_cnt, x="_Period_label", y="cnt", color=team_actual,
                            color_discrete_sequence=COLORS, barmode="stack",
                        )
                        fig_pt = chart_layout(fig_pt, f"Breaks by Period × Team — {selected_rec}",
                                              "Period", "Count", height=340)
                        st.plotly_chart(fig_pt, use_container_width=True)

                    # 2. Period × Entity
                    entity_actual = col_map.get("Entity")
                    if entity_actual and entity_actual in rec_df.columns:
                        period_ent = dq_local(
                            f'SELECT "_Period_label", "{entity_actual}", COUNT(*) AS cnt '
                            f'FROM rec_tbl GROUP BY "_Period_label", "{entity_actual}" '
                            f'ORDER BY "_Period_label"',
                            rec_tbl=rec_df
                        )
                        fig_pe = px.bar(
                            period_ent, x="_Period_label", y="cnt", color=entity_actual,
                            color_discrete_sequence=COLORS, barmode="stack",
                        )
                        fig_pe = chart_layout(fig_pe, f"Breaks by Period × Entity — {selected_rec}",
                                              "Period", "Count", height=340)
                        st.plotly_chart(fig_pe, use_container_width=True)

                    # 3. Period × Type of Break
                    type_actual = col_map.get("Type of Break")
                    if type_actual and type_actual in rec_df.columns:
                        period_type = dq_local(
                            f'SELECT "_Period_label", "{type_actual}", COUNT(*) AS cnt '
                            f'FROM rec_tbl GROUP BY "_Period_label", "{type_actual}" '
                            f'ORDER BY "_Period_label"',
                            rec_tbl=rec_df
                        )
                        fig_ptype = px.bar(
                            period_type, x="_Period_label", y="cnt", color=type_actual,
                            color_discrete_sequence=COLORS, barmode="stack",
                        )
                        fig_ptype = chart_layout(fig_ptype, f"Breaks by Period × Type of Break — {selected_rec}",
                                                 "Period", "Count", height=340)
                        st.plotly_chart(fig_ptype, use_container_width=True)

                    # 4. Period × Asset Class
                    ac_actual = col_map.get("Asset Class")
                    if ac_actual and ac_actual in rec_df.columns:
                        period_ac = dq_local(
                            f'SELECT "_Period_label", "{ac_actual}", COUNT(*) AS cnt '
                            f'FROM rec_tbl GROUP BY "_Period_label", "{ac_actual}" '
                            f'ORDER BY "_Period_label"',
                            rec_tbl=rec_df
                        )
                        fig_pac = px.bar(
                            period_ac, x="_Period_label", y="cnt", color=ac_actual,
                            color_discrete_sequence=COLORS, barmode="stack",
                        )
                        fig_pac = chart_layout(fig_pac, f"Breaks by Period × Asset Class — {selected_rec}",
                                               "Period", "Count", height=340)
                        st.plotly_chart(fig_pac, use_container_width=True)

                    # 5. Amount breakdown (if available)
                    abs_actual = col_map.get("ABS GBP") or col_map.get("BREAK AMOUNT GBP")
                    if abs_actual and abs_actual in rec_df.columns:
                        period_amt = dq_local(
                            f'SELECT "_Period_label", SUM({safe_amt(abs_actual)}) AS total_amt '
                            f'FROM rec_tbl GROUP BY "_Period_label" ORDER BY "_Period_label"',
                            rec_tbl=rec_df
                        )
                        fig_amt = px.bar(
                            period_amt, x="_Period_label", y="total_amt",
                            color_discrete_sequence=[PRIMARY],
                        )
                        fig_amt.update_traces(
                            text=[format_short(v) for v in period_amt["total_amt"]],
                            textposition="outside"
                        )
                        fig_amt = chart_layout(fig_amt, f"ABS GBP by Period — {selected_rec}",
                                               "Period", "ABS GBP (£)", height=320)
                        st.plotly_chart(fig_amt, use_container_width=True)

                        if team_actual and team_actual in rec_df.columns:
                            period_amt_team = dq_local(
                                f'SELECT "_Period_label", "{team_actual}", '
                                f'SUM({safe_amt(abs_actual)}) AS total_amt '
                                f'FROM rec_tbl GROUP BY "_Period_label", "{team_actual}" '
                                f'ORDER BY "_Period_label"',
                                rec_tbl=rec_df
                            )
                            fig_at = px.bar(
                                period_amt_team, x="_Period_label", y="total_amt", color=team_actual,
                                color_discrete_sequence=COLORS, barmode="stack",
                            )
                            fig_at = chart_layout(fig_at, f"ABS GBP by Period × Team — {selected_rec}",
                                                  "Period", "ABS GBP (£)", height=340)
                            st.plotly_chart(fig_at, use_container_width=True)

                        if entity_actual and entity_actual in rec_df.columns:
                            period_amt_ent = dq_local(
                                f'SELECT "_Period_label", "{entity_actual}", '
                                f'SUM({safe_amt(abs_actual)}) AS total_amt '
                                f'FROM rec_tbl GROUP BY "_Period_label", "{entity_actual}" '
                                f'ORDER BY "_Period_label"',
                                rec_tbl=rec_df
                            )
                            fig_ae = px.bar(
                                period_amt_ent, x="_Period_label", y="total_amt", color=entity_actual,
                                color_discrete_sequence=COLORS, barmode="stack",
                            )
                            fig_ae = chart_layout(fig_ae, f"ABS GBP by Period × Entity — {selected_rec}",
                                                  "Period", "ABS GBP (£)", height=340)
                            st.plotly_chart(fig_ae, use_container_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        # No rec column — just show overall period trend
        if "_Period_label" in df_f.columns:
            period_cnt = df_f.groupby("_Period_label").size().reset_index(name="Count").sort_values("_Period_label")
            fig = px.bar(period_cnt, x="_Period_label", y="Count", color_discrete_sequence=[PRIMARY])
            fig = chart_layout(fig, "Break Count by Period", "Period", "Count", height=380)
            st.plotly_chart(fig, use_container_width=True)


# ── Tab: Amount Analysis ──────────────────────────────────────────────────────

def tab_amount_analysis(df_f: pd.DataFrame, col_map: dict) -> None:
    st.markdown("## 💷 Amount Analysis")

    abs_actual = col_map.get("ABS GBP")
    brk_actual = col_map.get("BREAK AMOUNT GBP")
    amt_col = abs_actual or brk_actual

    if not amt_col or amt_col not in df_f.columns:
        st.info("No amount column found. Ensure 'ABS GBP' or 'BREAK AMOUNT GBP' is present.")
        return

    total_amt = df_f[amt_col].abs().sum()
    avg_amt   = df_f[amt_col].abs().mean()
    max_amt   = df_f[amt_col].abs().max()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Total ABS Amount (£)", format_short(total_amt))
    with k2:
        kpi_card("Avg ABS Amount (£)", format_short(avg_amt))
    with k3:
        kpi_card("Max ABS Amount (£)", format_short(max_amt))
    with k4:
        kpi_card("Total Breaks", format_number(len(df_f)))

    st.markdown("---")

    # Amount by Period
    if "_Period_label" in df_f.columns:
        period_amt = dq_local(
            f'SELECT "_Period_label", SUM({safe_amt(amt_col)}) AS total_amt '
            f'FROM tbl GROUP BY "_Period_label" ORDER BY "_Period_label"',
            tbl=df_f
        )
        fig = px.bar(
            period_amt, x="_Period_label", y="total_amt",
            color_discrete_sequence=[PRIMARY],
            text=[format_short(v) for v in period_amt["total_amt"]],
        )
        fig.update_traces(textposition="outside")
        fig = chart_layout(fig, "Total ABS Amount (£) by Period", "Period", "ABS GBP (£)", height=360)
        st.plotly_chart(fig, use_container_width=True)

    # Amount by Rec Name
    rec_actual = col_map.get("Rec Name (as per Rec Cube)")
    if rec_actual and rec_actual in df_f.columns:
        rec_amt = (
            df_f.groupby(rec_actual)[amt_col]
            .apply(lambda x: x.abs().sum())
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        rec_amt.columns = [rec_actual, "Total ABS (£)"]
        fig2 = px.bar(
            rec_amt, x=rec_actual, y="Total ABS (£)",
            color_discrete_sequence=[PRIMARY],
            text=[format_short(v) for v in rec_amt["Total ABS (£)"]],
        )
        fig2.update_traces(textposition="outside")
        fig2 = chart_layout(fig2, "Top-10 Rec Names by ABS Amount (£)", rec_actual, "ABS GBP (£)", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # Amount by Team
    team_actual = col_map.get("Team")
    if team_actual and team_actual in df_f.columns:
        team_amt = (
            df_f.groupby(team_actual)[amt_col]
            .apply(lambda x: x.abs().sum())
            .sort_values(ascending=False)
            .reset_index()
        )
        team_amt.columns = [team_actual, "Total ABS (£)"]
        fig3 = px.bar(
            team_amt, x=team_actual, y="Total ABS (£)",
            color_discrete_sequence=COLORS[:len(team_amt)],
            text=[format_short(v) for v in team_amt["Total ABS (£)"]],
        )
        fig3.update_traces(textposition="outside")
        fig3 = chart_layout(fig3, "ABS Amount (£) by Team", team_actual, "ABS GBP (£)", height=360)
        st.plotly_chart(fig3, use_container_width=True)

    # Distribution histogram
    st.markdown("### Amount Distribution")
    amt_data = df_f[amt_col].dropna()
    fig4 = px.histogram(amt_data, nbins=50, color_discrete_sequence=[PRIMARY])
    fig4 = chart_layout(fig4, "Distribution of ABS Amounts (£)", "ABS Amount (£)", "Count", height=340)
    st.plotly_chart(fig4, use_container_width=True)


# ── Tab: Jira Factor Analysis ─────────────────────────────────────────────────

def tab_jira_factor_analysis(df_f: pd.DataFrame, col_map: dict) -> None:
    st.markdown("## 🔍 Jira Factor Analysis")

    jira_flag_actual = col_map.get("Jira Flag")
    jira_id_actual   = col_map.get("Jira ID")

    if not jira_flag_actual and not jira_id_actual:
        st.info("No Jira Flag or Jira ID column found.")
        return

    # Derive jira mask
    if jira_flag_actual and jira_flag_actual in df_f.columns:
        jira_mask = df_f[jira_flag_actual].astype(str).str.lower().isin(
            ["true", "yes", "y", "1", "x"])
    elif jira_id_actual and jira_id_actual in df_f.columns:
        jira_mask = df_f[jira_id_actual].notna() & (df_f[jira_id_actual].astype(str).str.strip() != "")
    else:
        st.info("Jira column found but has no usable data.")
        return

    with_jira = int(jira_mask.sum())
    without_jira = len(df_f) - with_jira
    jira_pct = with_jira / max(len(df_f), 1) * 100

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("Breaks with Jira", format_number(with_jira), f"{jira_pct:.1f}% of total")
    with k2:
        kpi_card("Breaks without Jira", format_number(without_jira))
    with k3:
        if "_Computed_Age_Days" in df_f.columns:
            avg_age_jira    = df_f[jira_mask]["_Computed_Age_Days"].mean()
            avg_age_nojira  = df_f[~jira_mask]["_Computed_Age_Days"].mean()
            kpi_card("Avg Age (Jira)", format_number(avg_age_jira, 1))

    st.markdown("---")
    st.markdown(
        '<div class="banner-jira">💡 Jira tickets indicate escalated breaks. '
        'High Jira % may signal systemic issues requiring operational review.</div>',
        unsafe_allow_html=True)

    # Pie chart
    pie_data = pd.DataFrame({
        "Category": ["With Jira", "Without Jira"],
        "Count": [with_jira, without_jira],
    })
    fig_pie = px.pie(pie_data, names="Category", values="Count",
                     color_discrete_sequence=[WARN, PRIMARY])
    fig_pie = chart_layout(fig_pie, "Jira Coverage", height=360)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Jira rate by team
    team_actual = col_map.get("Team")
    if team_actual and team_actual in df_f.columns:
        st.markdown("### Jira Rate by Team")
        team_jira = df_f.groupby(team_actual).apply(
            lambda g: pd.Series({
                "Total": len(g),
                "With Jira": int(jira_mask.loc[g.index].sum()),
            })
        ).reset_index()
        team_jira["Jira Rate %"] = team_jira["With Jira"] / team_jira["Total"].clip(lower=1) * 100
        fig_tj = px.bar(
            team_jira, x=team_actual, y="Jira Rate %",
            color_discrete_sequence=[WARN],
            text=[f"{v:.1f}%" for v in team_jira["Jira Rate %"]],
        )
        fig_tj.update_traces(textposition="outside")
        fig_tj = chart_layout(fig_tj, "Jira Rate (%) by Team", team_actual, "Jira Rate %", height=360)
        st.plotly_chart(fig_tj, use_container_width=True)

    # Jira rate by period
    if "_Period_label" in df_f.columns:
        st.markdown("### Jira Rate Trend by Period")
        df_jira_trend = df_f.copy()
        df_jira_trend["_has_jira"] = jira_mask.astype(int)
        period_jira = (
            df_jira_trend.groupby("_Period_label")
            .agg(total=("_has_jira","count"), with_jira=("_has_jira","sum"))
            .reset_index()
        )
        period_jira["Jira Rate %"] = period_jira["with_jira"] / period_jira["total"].clip(lower=1) * 100
        fig_pj = px.line(
            period_jira, x="_Period_label", y="Jira Rate %",
            markers=True, color_discrete_sequence=[WARN],
        )
        fig_pj = chart_layout(fig_pj, "Jira Rate (%) by Period", "Period", "Jira Rate %", height=340)
        st.plotly_chart(fig_pj, use_container_width=True)


# ── Tab: FP Thresholding ──────────────────────────────────────────────────────

def tab_fp_thresholding(df_f: pd.DataFrame, col_map: dict) -> None:
    st.markdown("## 🎯 False Positive Thresholding")

    st.markdown(
        '<div class="banner-fp">🎯 Statistical FP detection: segments where the latest period '
        'break count is within historical norms are flagged as False Positive candidates.</div>',
        unsafe_allow_html=True)

    if "_Period_label" not in df_f.columns:
        st.info("No period data available for FP analysis.")
        return

    period_order = sorted(df_f["_Period_label"].dropna().unique().tolist())
    if len(period_order) < 2:
        st.info("Need at least 2 periods for FP thresholding.")
        return

    latest_period = period_order[-1]
    hist_periods  = period_order[:-1]

    # Segment columns
    seg_col_keys = ["Rec Name (as per Rec Cube)", "Team", "Entity", "Asset Class"]
    seg_cols = [col_map[k] for k in seg_col_keys if k in col_map and col_map[k] in df_f.columns]

    if not seg_cols:
        st.info("No segment columns found for FP analysis.")
        return

    # Sidebar controls
    c1, c2, c3 = st.columns(3)
    with c1:
        mad_k = st.slider("MAD multiplier (k)", 0.5, 5.0, 2.0, 0.5,
                          help="Number of MADs from historical mean to flag as FP candidate")
    with c2:
        fp_thresh_pct = st.slider("Deviation % threshold", 5, 100, 30, 5,
                                  help="Max % deviation from historical mean to flag as High confidence FP")
    with c3:
        min_hist_count = st.slider("Min historical count", 1, 20, 3, 1,
                                   help="Segments with historical mean below this are skipped")

    # Build pivot table: rows = segments, cols = periods, values = count
    grp_cols = seg_cols + ["_Period_label"]
    counts = df_f.groupby(grp_cols).size().reset_index(name="cnt")
    pivot = counts.pivot_table(index=seg_cols, columns="_Period_label", values="cnt", fill_value=0)
    pivot = pivot.reset_index()

    # Ensure all periods present
    for p in period_order:
        if p not in pivot.columns:
            pivot[p] = 0

    # Vectorised FP computation (Change 9)
    if len(pivot) == 0:
        st.info("No segments meet the minimum history count.")
        return

    hist_data   = pivot[hist_periods].values.astype(float)
    latest_data = pivot[latest_period].values.astype(float)

    hist_mean_all = hist_data.mean(axis=1)
    hist_mad_all  = np.median(np.abs(hist_data - hist_mean_all[:, None]), axis=1)

    mask_min  = hist_mean_all >= min_hist_count
    pivot     = pivot[mask_min].reset_index(drop=True)
    hist_data = hist_data[mask_min]
    hist_mean_all = hist_mean_all[mask_min]
    hist_mad_all  = hist_mad_all[mask_min]
    latest_data   = latest_data[mask_min]

    if len(pivot) == 0:
        st.info("No segments meet the minimum history count. Lower the Min historical count or upload more periods.")
        return

    dev_pct   = np.abs(latest_data - hist_mean_all) / np.maximum(hist_mean_all, 1) * 100
    z_mad     = np.abs(latest_data - hist_mean_all) / np.maximum(hist_mad_all, 0.5)

    pivot["Historical Mean"]  = np.round(hist_mean_all, 1)
    pivot["Historical MAD"]   = np.round(hist_mad_all, 2)
    pivot["Latest Count"]     = latest_data.astype(int)
    pivot["Deviation %"]      = np.round(dev_pct, 1)
    pivot["MAD Z-Score"]      = np.round(z_mad, 2)

    conditions = [
        (dev_pct <= fp_thresh_pct) & (z_mad <= mad_k),
        dev_pct <= fp_thresh_pct * 2,
    ]
    pivot["FP Confidence"] = np.select(conditions, ["🟢 High", "🟡 Medium"],
                                       default="🔴 Low (not FP)")
    pivot["Confirm as FP"] = pivot["FP Confidence"] == "🟢 High"

    # Add per-period count columns
    for p in period_order:
        pivot[f"Count {p}"] = pivot.get(p, pd.Series(0, index=pivot.index)).fillna(0).astype(int)

    result_df = pivot.sort_values("FP Confidence").reset_index(drop=True)

    display_cols = seg_cols + [
        "Historical Mean", "Historical MAD", "Latest Count",
        "Deviation %", "MAD Z-Score", "FP Confidence", "Confirm as FP"
    ] + [f"Count {p}" for p in period_order]
    display_df = result_df[[c for c in display_cols if c in result_df.columns]]

    st.markdown(f"### FP Candidates — Latest Period: {latest_period}")
    high_count = int((result_df["FP Confidence"] == "🟢 High").sum())
    med_count  = int((result_df["FP Confidence"] == "🟡 Medium").sum())
    st.markdown(f"Found **{high_count}** High confidence and **{med_count}** Medium confidence FP candidates.")

    edited = st.data_editor(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confirm as FP": st.column_config.CheckboxColumn("Confirm as FP", default=False),
        },
        key="_fp_editor",
    )

    if st.button("Apply FP Confirmations to Filters", key="_fp_apply"):
        confirmed = edited[edited["Confirm as FP"] == True]
        if len(confirmed) > 0:
            fp_seg_keys = [tuple(row[c] for c in seg_cols) for _, row in confirmed.iterrows()]
            st.session_state["_fp_seg_keys"] = fp_seg_keys
            st.session_state["_fp_seg_cols"] = seg_cols
            st.success(f"✅ {len(fp_seg_keys)} FP segments confirmed. Enable 'Exclude Confirmed False Positives' in the sidebar to filter them out.")
        else:
            st.session_state["_fp_seg_keys"] = []
            st.session_state["_fp_seg_cols"] = []
            st.info("No FP confirmations selected.")

    # Download
    st.download_button(
        "📥 Download FP Analysis (CSV)",
        display_df.to_csv(index=False).encode("utf-8"),
        file_name="fp_thresholding.csv",
        mime="text/csv",
    )


# ── Tab: Period Comparison ────────────────────────────────────────────────────

def tab_period_comparison(df_f: pd.DataFrame, hist_df, col_map: dict) -> None:
    st.markdown("## 📊 Period Comparison — Latest vs Historical")

    if hist_df is None or len(hist_df) == 0:
        st.info(
            "📭 No historical data found in cache yet. "
            "Upload additional period files and this tab will automatically "
            "compare your latest upload against prior cached periods."
        )
        return

    abs_col   = col_map.get("ABS GBP") or col_map.get("BREAK AMOUNT GBP")
    team_col  = col_map.get("Team")
    rec_col   = col_map.get("Rec Name (as per Rec Cube)")

    # Determine latest period in current upload vs history periods
    curr_periods = sorted(df_f["_Period_label"].dropna().unique().tolist()) if "_Period_label" in df_f.columns else []
    hist_periods_labels = sorted(hist_df["_Period_label"].dropna().unique().tolist()) if "_Period_label" in hist_df.columns else []
    latest_label = curr_periods[-1] if curr_periods else "Current Upload"

    st.markdown(
        f'<div class="banner-info">📊 Comparing <b>{latest_label}</b> vs '
        f'historical periods: <b>{", ".join(hist_periods_labels)}</b></div>',
        unsafe_allow_html=True
    )

    # ── KPI Comparison ────────────────────────────────────────────────────────
    n_curr = len(df_f)
    n_hist_periods = max(len(hist_periods_labels), 1)
    n_hist_avg = len(hist_df) / n_hist_periods

    mom = safe_mom_pct(n_curr, n_hist_avg)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Latest Breaks", format_number(n_curr),
                 f"Hist avg: {format_number(round(n_hist_avg))}")
    with k2:
        if abs_col and abs_col in df_f.columns and abs_col in hist_df.columns:
            curr_amt = float(df_f[abs_col].abs().sum())
            hist_amt = float(hist_df[abs_col].abs().sum()) / n_hist_periods
            amt_mom  = safe_mom_pct(curr_amt, hist_amt)
            kpi_card("Latest ABS GBP", format_short(curr_amt),
                     f"Hist avg: {format_short(hist_amt)}",
                     invert=True, warn=(amt_mom is not None and amt_mom > 20))
    with k3:
        if "_Computed_Age_Days" in df_f.columns and "_Computed_Age_Days" in hist_df.columns:
            curr_age = float(df_f["_Computed_Age_Days"].mean())
            hist_age = float(hist_df["_Computed_Age_Days"].mean())
            age_mom  = safe_mom_pct(curr_age, hist_age)
            kpi_card("Latest Avg Age Days", format_number(curr_age, 1),
                     f"Hist avg: {format_number(hist_age, 1)}",
                     invert=True, warn=(age_mom is not None and age_mom > 15))
    with k4:
        kpi_card("Historical Periods", str(len(hist_periods_labels)),
                 "vs 1 current")

    st.markdown("---")

    # ── Combined period trend ─────────────────────────────────────────────────
    st.markdown("### Break Count Trend — All Periods (Historical + Current)")
    trend_frames = []
    if "_Period_label" in hist_df.columns:
        h_trend = hist_df.groupby("_Period_label").size().reset_index(name="cnt")
        h_trend["source"] = "Historical"
        trend_frames.append(h_trend)
    if "_Period_label" in df_f.columns:
        c_trend = df_f.groupby("_Period_label").size().reset_index(name="cnt")
        c_trend["source"] = "Current Upload"
        trend_frames.append(c_trend)

    if trend_frames:
        all_trend = pd.concat(trend_frames, ignore_index=True)
        all_trend = all_trend.sort_values("_Period_label")
        fig = px.bar(all_trend, x="_Period_label", y="cnt", color="source",
                     color_discrete_map={"Historical": COLORS[1], "Current Upload": PRIMARY},
                     barmode="group")
        fig = chart_layout(fig, "Break Count by Period (All Data)", "Period", "Count", height=360)
        st.plotly_chart(fig, use_container_width=True)

    # ── Waterfall charts ──────────────────────────────────────────────────────
    st.markdown("### Delta Analysis — Current vs Historical Average")

    def make_waterfall_chart(dim_col, title, metric="count"):
        if dim_col not in df_f.columns or dim_col not in hist_df.columns:
            return None
        if metric == "count":
            curr_agg = df_f.groupby(dim_col).size()
            hist_agg = hist_df.groupby(dim_col).size() / n_hist_periods
        else:
            a = abs_col
            if not a or a not in df_f.columns:
                return None
            curr_agg = df_f.groupby(dim_col)[a].apply(lambda x: x.abs().sum())
            hist_agg = hist_df.groupby(dim_col)[a].apply(lambda x: x.abs().sum()) / n_hist_periods

        all_dims = curr_agg.index.union(hist_agg.index)
        curr_agg = curr_agg.reindex(all_dims, fill_value=0)
        hist_agg = hist_agg.reindex(all_dims, fill_value=0)
        delta = (curr_agg - hist_agg).sort_values()

        # Keep top/bottom 10 for readability
        if len(delta) > 20:
            delta = pd.concat([delta.head(10), delta.tail(10)]).drop_duplicates()

        colors = [WARN if v > 0 else PRIMARY for v in delta.values]
        fig = go.Figure(go.Bar(
            x=delta.index.astype(str), y=delta.values,
            marker_color=colors,
            text=[f"{'+' if v > 0 else ''}{format_number(v) if metric == 'count' else format_short(v)}"
                  for v in delta.values],
            textposition="outside",
        ))
        y_label = "Δ Breaks" if metric == "count" else "Δ ABS GBP (£)"
        return chart_layout(fig, title, dim_col.replace('"', ''), y_label, height=380)

    w1, w2 = st.columns(2)
    with w1:
        if rec_col:
            fig = make_waterfall_chart(rec_col, "Break Count Delta by Rec Name", "count")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    with w2:
        if team_col:
            fig = make_waterfall_chart(team_col, "Break Count Delta by Team", "count")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    if abs_col and abs_col in df_f.columns:
        w3, w4 = st.columns(2)
        with w3:
            if rec_col:
                fig = make_waterfall_chart(rec_col, "ABS GBP Delta by Rec Name", "amount")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        with w4:
            if team_col:
                fig = make_waterfall_chart(team_col, "ABS GBP Delta by Team", "amount")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "📥 Download Historical Data (CSV)",
        hist_df.to_csv(index=False).encode("utf-8"),
        file_name="historical_periods.csv",
        mime="text/csv",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.title("📊 Accounting Control AI Platform")

    # Sidebar: file upload
    st.sidebar.markdown("## Upload Data")
    uploaded = st.sidebar.file_uploader(
        "Upload Excel or CSV file",
        type=["xlsx", "xls", "csv"],
        key="_file_uploader",
    )

    if st.sidebar.button("Reset Filters", key="_reset_btn"):
        _reset_filters()
        st.rerun()

    if uploaded is None:
        st.markdown(
            '<div class="banner-info">👆 Upload an Excel or CSV file in the sidebar to begin analysis.</div>',
            unsafe_allow_html=True)
        st.stop()

    raw_bytes = uploaded.read()
    result = run_pipeline(raw_bytes)

    df      = result["df"]
    col_map = result["col_map"]

    # Set overflow in session state (Change 3)
    st.session_state["_overflow"] = result["overflow_count"]

    if result.get("cached"):
        st.sidebar.markdown(
            '<div class="banner-info" style="font-size:.78rem;">⚡ Loaded from cache.</div>',
            unsafe_allow_html=True)

    # Load historical context for period comparison (Change 12)
    current_periods = df["_Period_label"].dropna().unique().tolist() if "_Period_label" in df.columns else []
    hist_df = load_historical_context(current_periods, result["fhash"])

    # Build filters and apply
    filters = build_sidebar_filters(df, col_map)
    df_f = apply_filters(df, filters)

    # Tabs (Change 13)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🧹 Data Quality",
        "📅 Ageing Validation",
        "📈 Break Counts + Drill-Down",
        "💷 Amount Analysis",
        "🔍 Jira Factor Analysis",
        "🎯 FP Thresholding",
        "📊 Period Comparison",
    ])

    with tab1:
        tab_data_quality(df, df_f, col_map)
    with tab2:
        tab_ageing_validation(df_f, col_map)
    with tab3:
        tab_break_counts(df_f, col_map)
    with tab4:
        tab_amount_analysis(df_f, col_map)
    with tab5:
        tab_jira_factor_analysis(df_f, col_map)
    with tab6:
        tab_fp_thresholding(df_f, col_map)
    with tab7:
        tab_period_comparison(df_f, hist_df, col_map)


if __name__ == "__main__":
    main()
