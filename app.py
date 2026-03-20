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

import concurrent.futures
import hashlib
import json
import math
import os
import re
import tempfile
import time
from datetime import date as _date
from io import BytesIO

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
    HAS_AGGRID = True
except ImportError:
    HAS_AGGRID = False

try:
    from jira import Jira as _JiraClient
    HAS_JIRA = True
except ImportError:
    HAS_JIRA = False

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
  .jira-banner{{background:#E3F2FD;border-left:4px solid #1565C0;padding:.6rem 1rem;
      border-radius:6px;margin-bottom:1rem;font-size:.9rem;}}
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
        gb.configure_default_column(resizable=True, sortable=True, filter=True, floatingFilter=True)
        gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=20)
        _apply_computed_headers(gb, df.columns.tolist())
        AgGrid(df, gridOptions=gb.build(), height=height, key=key,
               allow_unsafe_jscode=True, theme="streamlit",
               update_mode=GridUpdateMode.NO_UPDATE)
    else:
        st.dataframe(df, width='stretch', height=height)


# ── Computed-column header styling ────────────────────────────────────────────
_COMPUTED_HEADER_COLS = frozenset({
    # Summary / Ageing aggregates
    "Break Count", "Avg Age Days", "Max Age Days",
    ">90d", ">90 Day %", ">90 Day Breaks", ">180d", ">180 Day Breaks", ">365d", ">365 Day Breaks",
    "Total ABS GBP", "Avg GBP / Break", "Unique Jira Refs",
    # MoM analytics
    "MoM Δ", "MoM Δ %",
    # Jira API enrichment
    "Jira Summary", "Assignee", "Reporter", "Status", "Created Date",
    # FP Thresholding priority outputs
    "Priority", "Latest ABS GBP", "Hist Avg ABS GBP", "vs Hist Avg %", "Trend",
    "Latest Break Count", "Hist Avg Break Count", "vs Hist Count %", "Count Trend",
    "Tag for Review",
    # Derived/renamed aggregates
    "Top Jira", "Jira Desc",
    # Ageing detail table renames
    "Age Days", "Ageing Bucket", "Period",
})
_PERIOD_RE     = re.compile(r'^\d{4}-\d{2}$')
_ABS_CNT_RE    = re.compile(r'^(ABS|Cnt) \d{4}-\d{2}$')
_YELLOW_HEADER_CLASS = "computed-col-header"


def _is_computed_col(col: str) -> bool:
    return (col in _COMPUTED_HEADER_COLS
            or _PERIOD_RE.match(col) is not None
            or _ABS_CNT_RE.match(col) is not None)


def _apply_computed_headers(gb, df_cols) -> None:
    """Apply yellow header class to every computed/derived column in df_cols."""
    for col in df_cols:
        if _is_computed_col(col):
            gb.configure_column(col, headerClass=_YELLOW_HEADER_CLASS)


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
    # Core dimensions
    "Rec Name (as per Rec Cube)": ["rec name", "rec_name", "reconciliation name", "rec cube", "recs_cube"],
    "Team":                       ["team", "team name"],
    "Entity":                     ["entity", "legal entity", "entity name"],
    "Type of Break":              ["type of break", "break type", "break_type"],
    "Asset Class":                ["asset class", "asset_class", "assetclass"],
    "Account Group":              ["account group", "account_group", "acct group"],
    "Products Reconciled":        ["products reconciled", "products_reconciled"],
    # Time
    "Date":     ["date", "break date", "trade date", "value date"],
    "Period":   ["period", "reporting period"],
    "Age Days":      ["age days", "age_days", "days aged", "days old"],
    "Ageing Bucket": ["ageing bucket", "aging bucket", "age bucket", "age_bucket", "ageing_bucket"],
    # Amounts
    "ABS GBP":          ["abs gbp", "abs_gbp", "absolute gbp"],
    "BREAK AMOUNT GBP": ["break amount gbp", "break_amount_gbp", "amount gbp", "gbp amount"],
    "BREAK AMOUNT CCY": ["break amount ccy", "break_amount_ccy"],
    "Threshold":        ["threshold"],
    "ABS GBP GT 1MN":   ["abs gbp(greater than 1mn)", "abs gbp (greater than 1mn)", "abs gbp>1mn"],
    # Jira & issue tracking
    "Jira Reference":     ["jira reference", "jira ref", "jira_reference", "jira_ref"],
    "Jira Desc":          ["jira desc", "jira description", "jira_desc"],
    "System to be Fixed": ["system to be fixed", "system_to_be_fixed", "system fix"],
    "ISSUE CATEGORY":     ["issue category", "issue_category"],
    "ISSUE CATEGORY2":    ["issue category2", "issue_category2"],
    "JIRA PRIORITY":      ["jira priority", "jira_priority"],
    "EPIC":               ["epic"],
    "EPIC DESC":          ["epic desc", "epic_desc", "epic description"],
    "High Level Product": ["high level product", "high_level_product"],
    "FIX REQUIRED":       ["fix required", "fix_required"],
    "ISSUE RAG RATING":   ["issue rag rating", "rag rating", "rag_rating"],
    # Classification
    "True/Systemic Breaks":  ["true/systemic", "true systemic", "systemic"],
    "Journals Posted":       ["journals posted", "journals_posted"],
    "Thematic":              ["thematic"],
    "Type of issue":         ["type of issue", "issue type"],
    "Action":                ["action"],
    "Root Cause identified": ["root cause", "root_cause"],
    "B/S Cert":              ["b/s cert", "bs cert"],
    # Trade fields
    "TRADE REF":      ["trade ref", "trade_ref"],
    "TRADE CCY":      ["trade ccy", "trade_ccy"],
    "RECS_CUBE_NAME": ["recs_cube_name", "recs cube name"],
    # Misc
    "Comments":      ["comments", "comment", "notes", "note"],
    "_FP_Confirmed": ["_fp_confirmed", "fp confirmed", "false positive confirmed"],
}


def fuzzy_match(col_name: str, candidates: list[str]) -> bool:
    cn = str(col_name).lower().strip()
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

    # Parse raw bytes — detect Parquet magic bytes (PAR1) first for fast loading
    if raw_bytes[:4] == b"PAR1":
        try:
            df = pd.read_parquet(BytesIO(raw_bytes))
        except Exception as e:
            st.error(f"Cannot parse Parquet file: {e}")
            st.stop()
    else:
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
        fp_keys_v2 = st.session_state.get("_fp_seg_keys_v2", [])
        fp_col_v2  = st.session_state.get("_fp_seg_col_v2", "")
        if fp_keys_v2 and fp_col_v2:
            st.sidebar.caption(
                f"Will exclude {len(fp_keys_v2)} tagged segment(s) from **{fp_col_v2}**."
            )
        else:
            st.sidebar.caption("No segments tagged yet. Use the FP Thresholding tab to tag.")

    st.sidebar.markdown("---")

    # Show pending-changes indicator if live widget state differs from applied snapshot
    applied = st.session_state.get("_filters_applied", {})
    _pending = filters != applied
    if _pending:
        st.sidebar.warning("⚠️ Pending filter changes — click **Apply Filters** to update charts.")

    if st.sidebar.button("Apply Filters", key="_apply_btn", type="primary"):
        st.session_state["_filters_applied"] = filters
        st.rerun()

    return applied


def _reset_filters():
    for k in ["_filters_applied", "_fp_seg_keys", "_fp_seg_cols",
              "_drill_rec", "_overflow"]:
        if k in st.session_state:
            del st.session_state[k]


def clear_all_cache() -> int:
    """Delete all cached Parquet files and the period index from disk,
    then wipe all session state keys. Returns number of files deleted."""
    cache = _cache_dir()
    deleted = 0
    for fname in os.listdir(cache):
        fpath = os.path.join(cache, fname)
        try:
            os.remove(fpath)
            deleted += 1
        except Exception:
            pass
    for k in list(st.session_state.keys()):
        try:
            del st.session_state[k]
        except Exception:
            pass
    return deleted


def _fetch_one_issue(ref_str: str, url: str, email: str, token: str) -> tuple:
    """Fetch one Jira issue; per-thread connection (thread-safe). Retries on 429."""
    ref_str = ref_str.strip()
    _NF = {"Jira Summary": "Not Found", "Assignee": "Not Found",
           "Reporter": "Not Found", "Status": "Not Found", "Created Date": "Not Found"}
    _delays = [2, 4, 8]
    for attempt, _delay in enumerate([0] + _delays):
        if _delay:
            time.sleep(_delay)
        try:
            conn  = _JiraClient(options={"server": url}, basic_auth=(email, token))
            issue = conn.issue(ref_str, fields="summary,assignee,reporter,status,created")
            return ref_str, {
                "Jira Summary": issue.fields.summary or "—",
                "Assignee":     getattr(issue.fields.assignee, "displayName", "—")
                                if issue.fields.assignee else "—",
                "Reporter":     getattr(issue.fields.reporter, "displayName", "—")
                                if issue.fields.reporter else "—",
                "Status":       getattr(issue.fields.status, "name", "—")
                                if issue.fields.status else "—",
                "Created Date": str(issue.fields.created)[:10]
                                if issue.fields.created else "—",
            }
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                if attempt < len(_delays):
                    continue          # retry after sleep
                return ref_str, {**_NF, "Jira Summary": "Rate limited — try again later"}
            return ref_str, {**_NF, "Jira Summary": f"Error: {type(e).__name__}: {err[:120]}"}
    return ref_str, {**_NF, "Jira Summary": "Rate limited — try again later"}


def fetch_jira_metadata(ctrls_refs: list, url: str, email: str, token: str,
                        progress_bar=None) -> dict:
    """
    Parallel-fetch Jira metadata for pre-filtered CTRLS- refs only.
    Each worker creates its own Jira connection (thread-safe; no shared session).
    """
    if not ctrls_refs:
        return {}
    total, done, result = len(ctrls_refs), 0, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, total)) as pool:
        futures = {pool.submit(_fetch_one_issue, r, url, email, token): r
                   for r in ctrls_refs}
        for future in concurrent.futures.as_completed(futures):
            ref_str, meta = future.result()
            result[ref_str] = meta
            done += 1
            if progress_bar is not None:
                progress_bar.progress(done / total)
    return result


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    where = []
    # Build filter using pandas directly for simplicity and thread safety
    result = df.copy()
    for key, val in filters.items():
        if key == "_EXCL_FP":
            # Legacy multi-column tuple exclusion
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
            # New single-column exclusion from redesigned FP / Priority tab
            fp_keys_v2 = st.session_state.get("_fp_seg_keys_v2", [])
            fp_col_v2  = st.session_state.get("_fp_seg_col_v2", "")
            if fp_keys_v2 and fp_col_v2 and fp_col_v2 in result.columns:
                result = result[~result[fp_col_v2].astype(str).isin(
                    [str(v) for v in fp_keys_v2]
                )]
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
    st.dataframe(pd.DataFrame(mapping_rows), width='stretch', hide_index=True)

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
    try:
        st.dataframe(
            null_df,
            column_config={
                "Null %": st.column_config.ProgressColumn(
                    "Null %", format="%.1f%%", min_value=0, max_value=100,
                )
            },
            hide_index=True,
            use_container_width=True,
        )
    except Exception:
        st.dataframe(null_df, width='stretch', hide_index=True)

    overflow_count = st.session_state.get("_overflow", 0)
    if overflow_count and overflow_count > 0:
        st.markdown(
            f'<div class="banner-warn">⚠️ {overflow_count} amount value(s) exceed JavaScript safe integer '
            f'({MAX_JS_INT:,}) and may display imprecisely in charts.</div>',
            unsafe_allow_html=True)


def tab_quality_and_ageing(
    df: pd.DataFrame, df_f: pd.DataFrame, col_map: dict, hist_df=None
) -> None:
    tab_data_quality(df, df_f, col_map)
    st.markdown("---")
    tab_ageing_validation(df_f, col_map, hist_df)


# ── Tab: Ageing Validation ────────────────────────────────────────────────────

def tab_ageing_validation(df_f: pd.DataFrame, col_map: dict, hist_df=None) -> None:
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

    # ── Pass/Fail Row A: Threshold Compliance (≤90d = Pass, >90d = Fail) ──
    st.markdown("---")
    st.markdown("#### Threshold Compliance (SLA: ≤90 Days)")
    pass_count = total - old_count
    pass_rate  = pass_count / max(total, 1) * 100
    pa1, pa2, pa3 = st.columns(3)
    with pa1:
        kpi_card("✅ Pass (≤90d)", format_number(pass_count),
                 f"{pass_rate:.1f}% of total", warn=False)
    with pa2:
        kpi_card("❌ Fail (>90d)", format_number(old_count),
                 f"{old_pct:.1f}% of total", warn=old_pct > 20)
    with pa3:
        kpi_card("Pass Rate", f"{pass_rate:.1f}%",
                 "% breaks within 90-day SLA", warn=pass_rate < 80)
    st.caption("Pass = break aged ≤90 days (within SLA).  Fail = break aged >90 days (SLA breach).")

    # ── Pass/Fail Row B: Source Bucket vs Computed Bucket Accuracy ──
    bucket_col = col_map.get("Ageing Bucket")
    if bucket_col and bucket_col in df_f.columns:
        st.markdown("---")
        st.markdown("#### Source Bucket Accuracy (Source vs Computed)")
        src_norm  = df_f[bucket_col].astype(str).str.strip().str.lower()
        comp_norm = df_f["_Computed_Bucket"].astype(str).str.strip().str.lower()
        match_mask   = src_norm == comp_norm
        matched      = int(match_mask.sum())
        mismatched   = total - matched
        match_rate   = matched / max(total, 1) * 100
        mismatch_pct = mismatched / max(total, 1) * 100
        pb1, pb2, pb3 = st.columns(3)
        with pb1:
            kpi_card("✅ Matched (Pass)", format_number(matched),
                     f"{match_rate:.1f}% of total", warn=False)
        with pb2:
            kpi_card("❌ Mismatched (Fail)", format_number(mismatched),
                     f"{mismatch_pct:.1f}% of total", warn=mismatch_pct > 5)
        with pb3:
            kpi_card("Bucket Accuracy", f"{match_rate:.1f}%",
                     "Source bucket matches computed", warn=match_rate < 95)
        st.caption(
            "Compares the source system's Ageing Bucket column against the bucket computed from Age Days. "
            "A mismatch indicates a data quality issue in the source system's classification."
        )

    st.markdown("---")

    fig = px.bar(
        bucket_counts, x="Bucket", y="Count",
        color_discrete_sequence=[PRIMARY],
        text="Count",
    )
    fig = chart_layout(fig, "Break Count by Age Bucket", "Age Bucket", "Count", height=380)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width='stretch')

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
        st.plotly_chart(fig2, width='stretch')

    # Period trend of ageing — merge historical data if available
    if "_Period_label" in df_f.columns and "_Computed_Age_Days" in df_f.columns:
        st.markdown("### Avg Age Days by Period")
        _age_cols = ["_Period_label", "_Computed_Age_Days"]
        if (hist_df is not None and len(hist_df) > 0
                and "_Period_label" in hist_df.columns
                and "_Computed_Age_Days" in hist_df.columns):
            _age_src = pd.concat([hist_df[_age_cols], df_f[_age_cols]], ignore_index=True)
        else:
            _age_src = df_f[_age_cols]
        age_trend = (
            _age_src.groupby("_Period_label")["_Computed_Age_Days"]
            .mean()
            .reset_index()
            .sort_values("_Period_label")
        )
        age_trend.columns = ["Period", "Avg Age Days"]
        fig3 = px.line(age_trend, x="Period", y="Avg Age Days",
                       markers=True, color_discrete_sequence=[PRIMARY])
        fig3 = chart_layout(fig3, "Average Age Days Trend by Period", "Period", "Avg Age Days", height=340)
        st.plotly_chart(fig3, width='stretch')

    # ── Ageing Detail: breaks > 90 days ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Breaks Exceeding 90-Day Threshold")

    if "_Computed_Age_Days" in df_f.columns:
        _age_cols = []
        for _lbl, _key in [
            ("Jira Reference",     "Jira Reference"),
            ("Rec Name",           "Rec Name (as per Rec Cube)"),
            ("System to be Fixed", "System to be Fixed"),
            ("Team",               "Team"),
            ("Issue Category",     "Issue Category"),
        ]:
            _c = col_map.get(_key)
            if _c and _c in df_f.columns:
                _age_cols.append(_c)
        if "_Period_label" in df_f.columns:
            _age_cols.append("_Period_label")
        _age_cols += ["_Computed_Age_Days", "_Computed_Bucket"]
        _abs_col = col_map.get("ABS GBP") or col_map.get("BREAK AMOUNT GBP")
        if _abs_col and _abs_col in df_f.columns:
            _age_cols.append(_abs_col)

        _age_df = (
            df_f[[c for c in _age_cols if c in df_f.columns]]
            .rename(columns={
                "_Period_label":      "Period",
                "_Computed_Age_Days": "Age Days",
                "_Computed_Bucket":   "Ageing Bucket",
            })
        )
        _age_df = _age_df[_age_df["Age Days"] > 90].sort_values("Age Days", ascending=False)

        st.caption(f"**{len(_age_df):,}** break record(s) with Age Days > 90")
        render_grid(_age_df, height=420, key="ageing_detail_grid")
        st.download_button(
            "📥 Download Ageing Detail (CSV)",
            _age_df.to_csv(index=False).encode("utf-8"),
            file_name="ageing_breaks_over_90d.csv",
            mime="text/csv",
        )
    else:
        st.info("Age data unavailable — ensure 'Age Days' or 'Date' column is mapped.")


# ── Tab: Jira Factor Analysis ─────────────────────────────────────────────────

def tab_jira_factor_analysis(df: pd.DataFrame, col_map: dict, hist_df=None):
    """
    Factor Analysis using Jira Reference (Jira Desc as lookup attribute — not a dimension)
    and System to be Fixed.

    Attribute roles:
      Jira Reference      → GROUP BY dimension (primary key)
      Jira Desc           → Lookup attribute shown as column alongside Jira Reference
      System to be Fixed  → GROUP BY dimension (separate factor)
      Rec Name            → GROUP BY dimension; Account Group / Products as lookup attrs
      TRADE REF           → Row identifier — never grouped
    """
    st.markdown("## Jira Factor Analysis")
    st.markdown(
        '<div class="jira-banner">🔍 <b>Factor Analysis:</b> Jira Reference and System to be Fixed '
        'are the primary break driver dimensions. Jira Description is shown as a reference '
        'attribute alongside each ticket — not grouped separately as it is 1:1 with Jira Reference. '
        'Similarly, Account Group and Products are shown as lookup attributes on Rec Name.</div>',
        unsafe_allow_html=True)

    # ── Resolve columns ──
    jira_ref_col   = col_map.get("Jira Reference")
    jira_desc_col  = col_map.get("Jira Desc")
    system_col     = col_map.get("System to be Fixed")
    abs_col        = col_map.get("ABS GBP") or col_map.get("BREAK AMOUNT GBP")
    team_col       = col_map.get("Team")
    rec_col        = col_map.get("Rec Name (as per Rec Cube)")
    ac_col         = col_map.get("Asset Class")
    entity_col     = col_map.get("Entity")
    ts_col         = col_map.get("True/Systemic Breaks")
    acct_grp_col   = col_map.get("Account Group")
    products_col   = col_map.get("Products Reconciled")

    # ── Dimension selector — Jira Desc excluded (it is a lookup, not a dimension) ──
    factor_options = {}
    if jira_ref_col and jira_ref_col in df.columns:
        factor_options["Jira Reference"]     = jira_ref_col
    if system_col   and system_col   in df.columns:
        factor_options["System to be Fixed"] = system_col

    extra_dims = {k: v for k, v in {
        "Team":          team_col,
        "Rec Name":      rec_col,
        "Asset Class":   ac_col,
        "Entity":        entity_col,
        "True/Systemic": ts_col,
    }.items() if v and v in df.columns}

    all_dims = {**factor_options, **extra_dims}

    if not all_dims:
        st.info("No Jira or dimension columns found. Check that the file contains "
                "'Jira reference', 'JIRA DESC', 'SYSTEM TO BE FIXED'.")
        return

    # ── Coverage KPIs for Jira columns ──
    coverage_cols = {k: v for k, v in {
        "Jira Reference":     jira_ref_col,
        "Jira Description":   jira_desc_col,
        "System to be Fixed": system_col,
    }.items() if v and v in df.columns}

    if coverage_cols:
        st.markdown("### Jira Column Coverage")
        cov_cols_ui = st.columns(len(coverage_cols))
        for i, (label, col) in enumerate(coverage_cols.items()):
            filled = dq(f"""
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (
                           WHERE "{col}" IS NOT NULL
                           AND TRIM(CAST("{col}" AS VARCHAR))
                               NOT IN ('','nan','None','N/A','-')
                       ) AS populated
                FROM tbl
            """, df).iloc[0]
            pct = round(int(filled["populated"]) / max(int(filled["total"]), 1) * 100, 1)
            with cov_cols_ui[i]:
                kpi_card(f"{label} Coverage", f"{pct}%",
                         f"{format_number(filled['populated'])} / {format_number(filled['total'])} rows",
                         warn=(pct < 50))

    st.markdown("---")

    # ── Dimension selector ──
    selected_label = st.selectbox(
        "Select Factor Dimension",
        list(all_dims.keys()),
        key="jira_dim_sel",
        help="Jira Description is not listed here — it appears as a lookup column "
             "in the summary table when Jira Reference is selected.",
    )
    dim_col = all_dims[selected_label]

    # Determine period order from combined current + historical data
    _period_sources = [df] if "_Period_label" in df.columns else []
    if hist_df is not None and len(hist_df) > 0 and "_Period_label" in hist_df.columns:
        _period_sources.append(hist_df[["_Period_label"]])
    if _period_sources:
        _combined_periods = pd.concat(_period_sources, ignore_index=True)
        period_order = sorted(_combined_periods["_Period_label"].dropna().unique().tolist())
    else:
        period_order = []
    latest = period_order[-1] if period_order else "0"
    prev   = period_order[-2] if len(period_order) >= 2 else "0"

    # ─────────────────────────────────────────────────────────────────────
    # Helper: build a Jira Desc / System lookup dict for annotations
    # ─────────────────────────────────────────────────────────────────────
    def _desc_map() -> dict:
        """Jira Reference → Jira Desc  (first non-null per ref)."""
        if not (jira_desc_col and jira_desc_col in df.columns): return {}
        return (dq(f"""
            SELECT "{jira_ref_col}" AS ref,
                   FIRST("{jira_desc_col}") AS d
            FROM tbl
            WHERE "{jira_ref_col}" IS NOT NULL
            GROUP BY "{jira_ref_col}"
        """, df).set_index("ref")["d"].to_dict())

    def _short_label(ref: str, dmap: dict, max_chars: int = 38) -> str:
        desc = str(dmap.get(ref, ""))
        if not desc or desc in ("nan", "None", "N/A", "-"): return str(ref)
        return f"{ref}: {desc[:max_chars]}{'…' if len(desc) > max_chars else ''}"

    # ── Factor Summary Table ──
    st.markdown(f"### {selected_label} — Factor Summary Table")

    # Lookup attribute expressions per selected dimension
    lookup_exprs = ""
    if selected_label == "Jira Reference":
        if jira_desc_col and jira_desc_col in df.columns:
            lookup_exprs += f', FIRST("{jira_desc_col}") AS "Jira Description"'
        if system_col and system_col in df.columns:
            lookup_exprs += f', FIRST("{system_col}") AS "System to be Fixed"'
    elif selected_label == "System to be Fixed":
        if jira_ref_col and jira_ref_col in df.columns:
            lookup_exprs += f', COUNT(DISTINCT "{jira_ref_col}") AS "Unique Jira Refs"'
    elif selected_label == "Rec Name":
        if acct_grp_col and acct_grp_col in df.columns:
            lookup_exprs += f', FIRST("{acct_grp_col}") AS "Account Group"'
        if products_col and products_col in df.columns:
            lookup_exprs += f', FIRST("{products_col}") AS "Products Reconciled"'

    age_expr = (
        'ROUND(AVG(_Computed_Age_Days), 1) AS "Avg Age Days",'
        'MAX(_Computed_Age_Days)           AS "Max Age Days",'
        if "_Computed_Age_Days" in df.columns else ""
    )
    amount_expr = (
        f'ROUND(SUM(ABS("{abs_col}")), 0) AS "Total ABS GBP",'
        if abs_col and abs_col in df.columns else ""
    )
    over_expr = (
        'COUNT(*) FILTER (WHERE _Computed_Age_Days > 90)  AS ">90d",'
        'COUNT(*) FILTER (WHERE _Computed_Age_Days > 180) AS ">180d",'
        'COUNT(*) FILTER (WHERE _Computed_Age_Days > 365) AS ">365d",'
        if "_Computed_Age_Days" in df.columns else ""
    )
    # Build one column per period so all historical + current periods are visible
    if period_order:
        _p_safe = lambda p: p.replace("'", "''")
        period_expr = ",\n".join(
            f'COUNT(*) FILTER (WHERE _Period_label = \'{_p_safe(p)}\') AS "{p}"'
            for p in period_order
        )
    else:
        period_expr = '0 AS "No Period"'

    # Build combined source for summary: include hist data so Prev Period / Latest are populated
    if hist_df is not None and len(hist_df) > 0 and "_Period_label" in hist_df.columns:
        _summary_src = pd.concat([hist_df, df], ignore_index=True)
    else:
        _summary_src = df

    summary_df = dq(f"""
        SELECT
            "{dim_col}"  AS "{selected_label}",
            COUNT(*)     AS "Break Count"
            {lookup_exprs},
            {age_expr}
            {amount_expr}
            {over_expr}
            {period_expr}
        FROM tbl
        WHERE "{dim_col}" IS NOT NULL
          AND TRIM(CAST("{dim_col}" AS VARCHAR)) NOT IN ('','nan','None','N/A','-')
        GROUP BY "{dim_col}"
        ORDER BY "Break Count" DESC
    """, _summary_src)

    # ── Derived / enriched columns ──
    if "Total ABS GBP" in summary_df.columns:
        summary_df["Avg GBP / Break"] = (
            summary_df["Total ABS GBP"] /
            summary_df["Break Count"].replace(0, np.nan)
        ).round(0)
        summary_df["Total ABS GBP"]   = summary_df["Total ABS GBP"].apply(format_short)
        summary_df["Avg GBP / Break"] = summary_df["Avg GBP / Break"].apply(format_short)

    if ">90d" in summary_df.columns:
        summary_df[">90 Day %"] = (
            summary_df[">90d"] /
            summary_df["Break Count"].replace(0, np.nan) * 100
        ).round(1)

    # MoM Δ uses the two most recent periods (latest vs previous)
    if len(period_order) >= 2 and latest in summary_df.columns and prev in summary_df.columns:
        summary_df["MoM Δ"]   = summary_df[latest] - summary_df[prev]
        summary_df["MoM Δ %"] = (
            (summary_df["MoM Δ"] /
             summary_df[prev].replace(0, np.nan)) * 100
        ).round(1)

    # ── Live Jira Enrichment (incremental per-ref cache + parallel fetch) ──────────
    _jira_enriched_cols = []
    if selected_label == "Jira Reference" and HAS_JIRA:
        _jira_url   = st.session_state.get("_jira_url", "")
        _jira_email = st.session_state.get("_jira_email", "")
        _jira_token = st.session_state.get("_jira_token", "")
        if _jira_url and _jira_email and _jira_token:
            # Clear store when credentials change
            _cred_hash = hashlib.md5((_jira_url + _jira_email + _jira_token).encode()).hexdigest()[:12]
            if st.session_state.get("_jira_cred_hash") != _cred_hash:
                st.session_state["_jira_cred_hash"]  = _cred_hash
                st.session_state["_jira_meta_store"] = {}
            _store = st.session_state.setdefault("_jira_meta_store", {})

            _all_refs = sorted({str(r).strip() for r in summary_df[selected_label].tolist()
                                if str(r).strip() not in ("", "nan", "None")})
            _SKIP = {"Jira Summary": "Skip", "Assignee": "Skip", "Reporter": "Skip",
                     "Status": "Skip", "Created Date": "Skip"}

            # Mark non-CTRLS refs as Skip immediately (no API call ever)
            for _r in _all_refs:
                if "CTRLS-" not in _r.upper() and _r not in _store:
                    _store[_r] = _SKIP

            # Only fetch CTRLS- refs not yet in store
            _to_fetch = [r for r in _all_refs if "CTRLS-" in r.upper() and r not in _store]

            if _to_fetch:
                _pbar = st.progress(0.0, text=f"Fetching {len(_to_fetch)} new CTRLS- ref(s)…")
                _new  = fetch_jira_metadata(_to_fetch, _jira_url, _jira_email, _jira_token,
                                            progress_bar=_pbar)
                _store.update(_new)
                _pbar.empty()

            _jira_meta = {r: _store[r] for r in _all_refs if r in _store}
            _n_fetched = sum(1 for v in _jira_meta.values()
                             if v.get("Jira Summary") not in ("Skip", "Not Found", "")
                             and not str(v.get("Jira Summary", "")).startswith("Connect error"))
            _n_skip    = sum(1 for v in _jira_meta.values() if v.get("Jira Summary") == "Skip")
            st.caption(f"🔗 Jira: **{_n_fetched}** CTRLS- fetched · **{_n_skip}** skipped · "
                       f"**{len(_store)}** in store")
            _meta_df = (
                pd.DataFrame.from_dict(_jira_meta, orient="index")
                .reset_index()
                .rename(columns={"index": selected_label})
            )
            summary_df = summary_df.merge(_meta_df, on=selected_label, how="left")
            _jira_enriched_cols = ["Jira Summary", "Assignee", "Reporter", "Status", "Created Date"]
        else:
            st.caption("🔗 Enter Jira credentials in the sidebar (**🔗 Jira Integration**) to enable live enrichment.")

    if len(period_order) < 2:
        st.info("💡 **MoM Δ unavailable** — upload a historical file (sidebar) or ensure "
                "the dataset spans 2+ periods to enable month-over-month comparison.")

    # ── Column order: metadata, Break Count, all period columns oldest→latest, MoM, ageing, amounts ──
    col_order = [selected_label]
    for c in ["Jira Description", "System to be Fixed",
              "Jira Summary", "Assignee", "Reporter", "Status", "Created Date",
              "Account Group", "Products Reconciled", "Unique Jira Refs"]:
        if c in summary_df.columns: col_order.append(c)
    col_order.append("Break Count")
    # All period columns in chronological order
    for p in period_order:
        if p in summary_df.columns: col_order.append(p)
    for c in ["MoM Δ", "MoM Δ %",
              "Avg Age Days", "Max Age Days",
              ">90d", ">90 Day %", ">180d", ">365d",
              "Total ABS GBP", "Avg GBP / Break"]:
        if c in summary_df.columns: col_order.append(c)
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

    # ── Rename >90d etc. for display ──
    summary_df = summary_df.rename(columns={
        ">90d":  ">90 Day Breaks",
        ">180d": ">180 Day Breaks",
        ">365d": ">365 Day Breaks",
    })

    # ── AgGrid with rich conditional formatting ──
    if HAS_AGGRID:
        gb = GridOptionsBuilder.from_dataframe(summary_df)
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
        gb.configure_default_column(filter=True, sortable=True, resizable=True,
                                    wrapText=True, autoHeight=True, floatingFilter=True)
        gb.configure_column(selected_label, pinned="left", width=160)
        if "Jira Description" in summary_df.columns:
            gb.configure_column("Jira Description", width=300,
                                tooltipField="Jira Description")
        if "System to be Fixed" in summary_df.columns:
            gb.configure_column("System to be Fixed", width=200)
        if "Jira Summary" in summary_df.columns:
            gb.configure_column("Jira Summary", width=300, tooltipField="Jira Summary")
        if "Assignee" in summary_df.columns:
            gb.configure_column("Assignee", width=150)
        if "Reporter" in summary_df.columns:
            gb.configure_column("Reporter", width=150)
        if "Created Date" in summary_df.columns:
            gb.configure_column("Created Date", width=120)
        for c in ["Break Count"] + period_order:
            if c in summary_df.columns:
                gb.configure_column(c, width=110)
        # Apply yellow header styling first; cell-level styling below takes final precedence
        _apply_computed_headers(gb, summary_df.columns.tolist())
        if "Status" in summary_df.columns:
            status_cs = JsCode("""function(p){
                var v=(p.value||'').toLowerCase().trim();
                if(v==='done')
                    return {'backgroundColor':'#E8F5E9','color':'#1B5E20','fontWeight':'bold'};
                if(v==='to do'||v==='todo')
                    return {'backgroundColor':'#F5F5F5','color':'#424242','fontWeight':'bold'};
                if(v==='backlog')
                    return {'backgroundColor':'#FFEBEE','color':'#B71C1C','fontWeight':'bold'};
                if(['in analysis','in development','in testing'].indexOf(v)>=0)
                    return {'backgroundColor':'#E3F2FD','color':'#0D47A1','fontWeight':'bold'};
                return {};}""")
            gb.configure_column("Status", cellStyle=status_cs, width=110)
        if "MoM Δ" in summary_df.columns:
            mom_cs = JsCode("""function(p){
                var v=p.value;
                if(v > 0) return {'backgroundColor':'#FDECEA','color':'#A84B2F','fontWeight':'bold'};
                if(v < 0) return {'backgroundColor':'#E8F5E9','color':'#01696F','fontWeight':'bold'};
                return {};}""")
            gb.configure_column("MoM Δ",   cellStyle=mom_cs, width=90)
            gb.configure_column("MoM Δ %", cellStyle=mom_cs, width=100)
        if ">90 Day %" in summary_df.columns:
            risk_cs = JsCode("""function(p){
                var v=parseFloat(p.value);
                if(v>=50) return {'backgroundColor':'#FDECEA','color':'#A84B2F','fontWeight':'bold'};
                if(v>=25) return {'backgroundColor':'#FFF8E1','color':'#7A4000'};
                return {};}""")
            gb.configure_column(">90 Day %", cellStyle=risk_cs, width=105)
        AgGrid(summary_df, gridOptions=gb.build(), height=500,
               theme="streamlit", allow_unsafe_jscode=True, key="jira_summary_grid",
               update_mode=GridUpdateMode.NO_UPDATE)
    else:
        st.dataframe(summary_df, height=500, width='stretch')

    # ── Download button right below the table ──
    st.download_button(
        f"📥 Download {selected_label} Factor Summary (CSV)",
        summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"factor_{selected_label.replace(' ','_').lower()}.csv",
        mime="text/csv",
    )
    # Raw underlying rows for current filtered upload (no internal _ columns)
    _raw_dl = df.rename(columns={"_Period_label": "Period"})
    _raw_dl = _raw_dl[[c for c in _raw_dl.columns if not c.startswith("_")]]
    st.download_button(
        "📥 Download Raw Breaks — Latest File (CSV)",
        _raw_dl.to_csv(index=False).encode("utf-8"),
        file_name="raw_breaks_latest.csv",
        mime="text/csv",
    )

    # ── Drill-Down: select a dimension value and break it by a secondary dim ──
    st.markdown("---")
    st.markdown(f"### Drill-Down: {selected_label} → Secondary Breakdown")

    _type_col = col_map.get("Type of Break")
    _secondary_opts = {}
    for _lbl, _col in {
        "Team":          team_col,
        "Rec Name":      rec_col,
        "Entity":        entity_col,
        "Asset Class":   ac_col,
        "Type of Break": _type_col,
        "True/Systemic": ts_col,
    }.items():
        if _col and _col != dim_col and _col in _summary_src.columns:
            _secondary_opts[_lbl] = _col

    if not _secondary_opts:
        st.info("No secondary breakdown columns available for the current dimension.")
    else:
        _dim_vals = sorted(
            v for v in _summary_src[dim_col].dropna().astype(str).unique()
            if v not in ("", "nan", "None", "N/A", "-")
        )
        if not _dim_vals:
            st.info("No values found to drill into.")
        else:
            dd1, dd2, dd3 = st.columns([3, 2, 2])
            with dd1:
                _sel_val = st.selectbox(
                    f"Select {selected_label}:",
                    _dim_vals, key="jira_drill_val",
                )
            with dd2:
                _drill_dim_lbl = st.selectbox(
                    "Breakdown by", list(_secondary_opts.keys()),
                    key="jira_drill_dim",
                )
            with dd3:
                _drill_metric = st.radio(
                    "Metric", ["Count", "ABS GBP"], horizontal=True,
                    key="jira_drill_metric",
                )

            _drill_dim_col = _secondary_opts[_drill_dim_lbl]
            _drill_src = _summary_src[
                _summary_src[dim_col].astype(str) == _sel_val
            ].copy()

            _drill_dl = _drill_src.rename(columns={"_Period_label": "Period"})
            _drill_dl = _drill_dl[[c for c in _drill_dl.columns if not c.startswith("_")]]
            st.download_button(
                f"📥 Download Breaks for {_sel_val} (CSV)",
                _drill_dl.to_csv(index=False).encode("utf-8"),
                file_name=f"breaks_{str(_sel_val).replace('/', '_')[:40]}.csv",
                mime="text/csv",
            )

            if len(_drill_src) == 0:
                st.info(f"No data found for {selected_label} = {_sel_val}.")
            else:
                st.markdown(
                    f'<div class="banner-drill">🔬 Drilling into: <b>{_sel_val}</b> '
                    f'— breakdown by <b>{_drill_dim_lbl}</b></div>',
                    unsafe_allow_html=True,
                )
                _fig_dd = None
                if _drill_metric == "Count":
                    _drill_trend = dq_local(
                        f'SELECT "_Period_label", "{_drill_dim_col}", COUNT(*) AS cnt '
                        f'FROM tbl WHERE "_Period_label" IS NOT NULL '
                        f'GROUP BY "_Period_label", "{_drill_dim_col}" '
                        f'ORDER BY "_Period_label"',
                        tbl=_drill_src,
                    )
                    _fig_dd = px.bar(
                        _drill_trend, x="_Period_label", y="cnt",
                        color=_drill_dim_col,
                        color_discrete_sequence=COLORS, barmode="stack",
                    )
                    _fig_dd = chart_layout(
                        _fig_dd,
                        f"{_sel_val} — Breaks by Period × {_drill_dim_lbl}",
                        "Period", "Break Count", height=420,
                    )
                else:
                    if abs_col and abs_col in _drill_src.columns:
                        _drill_trend = dq_local(
                            f'SELECT "_Period_label", "{_drill_dim_col}", '
                            f'SUM(ABS("{abs_col}")) AS total_abs '
                            f'FROM tbl WHERE "_Period_label" IS NOT NULL '
                            f'GROUP BY "_Period_label", "{_drill_dim_col}" '
                            f'ORDER BY "_Period_label"',
                            tbl=_drill_src,
                        )
                        _fig_dd = px.bar(
                            _drill_trend, x="_Period_label", y="total_abs",
                            color=_drill_dim_col,
                            color_discrete_sequence=COLORS, barmode="stack",
                        )
                        _fig_dd = chart_layout(
                            _fig_dd,
                            f"{_sel_val} — ABS GBP by Period × {_drill_dim_lbl}",
                            "Period", "ABS GBP (£)", height=420,
                        )
                    else:
                        st.info("No ABS GBP column available for Amount metric.")

                if _fig_dd is not None:
                    st.plotly_chart(_fig_dd, width='stretch')

    st.markdown("---")

    # ─────────────────────────────────────────────────────────────────────
    # Visual panels — 4 charts in 2×2 grid
    # ─────────────────────────────────────────────────────────────────────
    dmap = _desc_map() if selected_label == "Jira Reference" else {}

    row1a, row1b = st.columns(2)

    # ── Chart 1: Top 15 by Break Count ──
    with row1a:
        top_cnt = dq(f"""
            SELECT "{dim_col}" AS factor, COUNT(*) AS cnt
            FROM tbl
            WHERE "{dim_col}" IS NOT NULL
              AND TRIM(CAST("{dim_col}" AS VARCHAR)) NOT IN ('','nan','None','N/A','-')
            GROUP BY "{dim_col}" ORDER BY cnt DESC LIMIT 15
        """, df).sort_values("cnt", ascending=True)

        if dmap:
            top_cnt["hover_desc"] = top_cnt["factor"].map(dmap).fillna("")
            fig = go.Figure(go.Bar(
                x=top_cnt["cnt"], y=top_cnt["factor"],
                orientation="h", marker_color=PRIMARY,
                text=top_cnt["cnt"], textposition="outside",
                customdata=top_cnt["hover_desc"],
                hovertemplate="<b>%{y}</b><br>%{customdata}<br>Breaks: %{x}<extra></extra>",
            ))
        else:
            fig = px.bar(top_cnt, x="cnt", y="factor", orientation="h",
                         color_discrete_sequence=[PRIMARY], text="cnt")
            fig.update_traces(textposition="outside")
        fig = chart_layout(fig, f"Top 15 {selected_label} by Break Count",
                           "Break Count", "", height=max(360, len(top_cnt) * 32))
        st.plotly_chart(fig, width='stretch')

    # ── Chart 2: Risk Matrix — Age vs Break Count (bubble = ABS GBP) ──
    with row1b:
        if "_Computed_Age_Days" in df.columns:
            _abs_expr = (
                f', ROUND(SUM(ABS("{abs_col}")), 0) AS total_abs'
                if abs_col and abs_col in df.columns else ""
            )
            risk_matrix = dq(f"""
                SELECT "{dim_col}" AS factor,
                       ROUND(AVG(_Computed_Age_Days), 1) AS avg_age,
                       COUNT(*) AS brk_cnt
                       {_abs_expr}
                FROM tbl
                WHERE "{dim_col}" IS NOT NULL
                  AND TRIM(CAST("{dim_col}" AS VARCHAR)) NOT IN ('','nan','None','N/A','-')
                  AND _Computed_Age_Days IS NOT NULL
                GROUP BY "{dim_col}" ORDER BY avg_age DESC LIMIT 20
            """, df)

            if len(risk_matrix) > 0:
                bubble_colors = [
                    WARN      if v > 180 else
                    COLORS[8] if v > 90  else
                    COLORS[0]
                    for v in risk_matrix["avg_age"]
                ]
                if "total_abs" in risk_matrix.columns:
                    max_abs = risk_matrix["total_abs"].replace(0, np.nan).max() or 1
                    bubble_sizes = (
                        (risk_matrix["total_abs"].fillna(0) / max_abs * 50 + 10)
                        .clip(10, 60).tolist()
                    )
                    size_label = risk_matrix["total_abs"].apply(format_short)
                else:
                    bubble_sizes = [20] * len(risk_matrix)
                    size_label = pd.Series(["N/A"] * len(risk_matrix))

                if dmap:
                    risk_matrix["hover_desc"] = risk_matrix["factor"].map(dmap).fillna("")
                    htext = (
                        "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
                        "Avg Age: %{x}d<br>Breaks: %{y}<br>"
                        "ABS GBP: %{customdata[2]}<extra></extra>"
                    )
                    cdata = list(zip(
                        risk_matrix["factor"],
                        risk_matrix["hover_desc"],
                        size_label,
                    ))
                else:
                    htext = (
                        "<b>%{customdata[0]}</b><br>"
                        "Avg Age: %{x}d<br>Breaks: %{y}<br>"
                        "ABS GBP: %{customdata[1]}<extra></extra>"
                    )
                    cdata = list(zip(risk_matrix["factor"], size_label))

                fig = go.Figure(go.Scatter(
                    x=risk_matrix["avg_age"],
                    y=risk_matrix["brk_cnt"],
                    mode="markers+text",
                    text=risk_matrix["factor"].apply(
                        lambda v: (str(v)[:18] + "…") if len(str(v)) > 18 else str(v)
                    ),
                    textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(
                        size=bubble_sizes,
                        color=bubble_colors,
                        opacity=0.75,
                        line=dict(width=1, color="white"),
                    ),
                    customdata=cdata,
                    hovertemplate=htext,
                ))
                fig.add_vline(x=90,  line_dash="dash", line_color=COLORS[8], line_width=1.2,
                              annotation_text="90d",  annotation_position="top right")
                fig.add_vline(x=180, line_dash="dash", line_color=WARN,      line_width=1.2,
                              annotation_text="180d", annotation_position="top right")
                fig = chart_layout(fig,
                    f"Risk Matrix: {selected_label} — Age vs Break Count",
                    "Avg Age Days", "Break Count", height=420)
                fig.update_xaxes(type="linear")
                st.plotly_chart(fig, width='stretch')

    if abs_col and abs_col in df.columns:
        row2a, row2b = st.columns(2)

        # ── Chart 3: Top 15 by ABS GBP ──
        with row2a:
            top_amt = dq(f"""
                SELECT "{dim_col}" AS factor, SUM(ABS("{abs_col}")) AS total
                FROM tbl
                WHERE "{dim_col}" IS NOT NULL
                  AND TRIM(CAST("{dim_col}" AS VARCHAR)) NOT IN ('','nan','None','N/A','-')
                GROUP BY "{dim_col}" ORDER BY total DESC LIMIT 15
            """, df).sort_values("total", ascending=True)
            top_amt["label"] = top_amt["total"].apply(format_short)
            if dmap:
                top_amt["hover_desc"] = top_amt["factor"].map(dmap).fillna("")
                fig = go.Figure(go.Bar(
                    x=top_amt["total"], y=top_amt["factor"],
                    orientation="h", marker_color=COLORS[2],
                    text=top_amt["label"], textposition="outside",
                    customdata=top_amt["hover_desc"],
                    hovertemplate="<b>%{y}</b><br>%{customdata}<br>ABS GBP: %{text}<extra></extra>",
                ))
            else:
                fig = px.bar(top_amt, x="total", y="factor", orientation="h",
                             color_discrete_sequence=[COLORS[2]], text="label")
                fig.update_traces(textposition="outside")
            fig = chart_layout(fig, f"Top 15 {selected_label} by Total ABS GBP",
                               "GBP (£)", "", height=max(360, len(top_amt) * 32))
            st.plotly_chart(fig, width='stretch')

        # ── Chart 4: Top 15 by % Breaks > 90 Days ──
        with row2b:
            if "_Computed_Age_Days" in df.columns:
                risk_df = dq(f"""
                    SELECT "{dim_col}" AS factor,
                           COUNT(*) AS total,
                           COUNT(*) FILTER (WHERE _Computed_Age_Days > 90) AS over90
                    FROM tbl
                    WHERE "{dim_col}" IS NOT NULL
                      AND TRIM(CAST("{dim_col}" AS VARCHAR)) NOT IN ('','nan','None','N/A','-')
                      AND _Computed_Age_Days IS NOT NULL
                    GROUP BY "{dim_col}" ORDER BY over90 DESC LIMIT 15
                """, df)
                risk_df[">90 Day %"] = (
                    risk_df["over90"] / risk_df["total"].replace(0, np.nan) * 100
                ).round(1)
                risk_df = risk_df.sort_values(">90 Day %", ascending=True)
                risk_colors = [
                    WARN      if v >= 50 else
                    COLORS[8] if v >= 25 else
                    COLORS[0]
                    for v in risk_df[">90 Day %"]
                ]
                if dmap:
                    risk_df["hover_desc"] = risk_df["factor"].map(dmap).fillna("")
                    fig = go.Figure(go.Bar(
                        x=risk_df[">90 Day %"], y=risk_df["factor"],
                        orientation="h", marker_color=risk_colors,
                        text=risk_df[">90 Day %"].astype(str) + "%",
                        textposition="outside",
                        customdata=risk_df["hover_desc"],
                        hovertemplate="<b>%{y}</b><br>%{customdata}<br>>90d: %{x}%<extra></extra>",
                    ))
                else:
                    fig = go.Figure(go.Bar(
                        x=risk_df[">90 Day %"], y=risk_df["factor"],
                        orientation="h", marker_color=risk_colors,
                        text=risk_df[">90 Day %"].astype(str) + "%",
                        textposition="outside",
                    ))
                fig.add_vline(x=25, line_dash="dash", line_color=COLORS[8], line_width=1.2,
                              annotation_text="25%", annotation_position="top right")
                fig.add_vline(x=50, line_dash="dash", line_color=WARN,      line_width=1.2,
                              annotation_text="50%", annotation_position="top right")
                fig = chart_layout(fig, f"Top 15 {selected_label} — % Breaks > 90 Days",
                                   "% Breaks > 90 Days", "",
                                   height=max(360, len(risk_df) * 32))
                st.plotly_chart(fig, width='stretch')

    # ─────────────────────────────────────────────────────────────────────
    # Trend over periods — Top 5, legend annotated with Jira Desc
    # ─────────────────────────────────────────────────────────────────────
    if "_Period_label" in _summary_src.columns and period_order:
        st.markdown(f"### {selected_label} — Break Count Trend Over Periods (Top 5)")
        top5 = dq(f"""
            SELECT "{dim_col}" AS factor FROM tbl
            WHERE "{dim_col}" IS NOT NULL
              AND TRIM(CAST("{dim_col}" AS VARCHAR)) NOT IN ('','nan','None','N/A','-')
            GROUP BY "{dim_col}" ORDER BY COUNT(*) DESC LIMIT 5
        """, _summary_src)["factor"].tolist()
        top5_str = ", ".join(f"'{str(v).replace(chr(39), chr(39)*2)}'" for v in top5)
        trend_df = dq(f"""
            SELECT "{dim_col}" AS factor, _Period_label AS period, COUNT(*) AS cnt
            FROM tbl WHERE "{dim_col}" IN ({top5_str})
            GROUP BY "{dim_col}", _Period_label ORDER BY _Period_label
        """, _summary_src)
        if dmap:
            trend_df["factor_label"] = trend_df["factor"].apply(
                lambda x: _short_label(x, dmap, max_chars=40))
        else:
            trend_df["factor_label"] = trend_df["factor"]
        fig = px.line(trend_df, x="period", y="cnt", color="factor_label",
                      color_discrete_sequence=COLORS, markers=True)
        fig = chart_layout(fig,
            f"Top 5 {selected_label} — Break Count Trend Over Periods",
            "Period", "Break Count")
        st.plotly_chart(fig, width='stretch')

    # ─────────────────────────────────────────────────────────────────────
    # MoM Change — bar chart with hover showing prev/latest counts
    # ─────────────────────────────────────────────────────────────────────
    if len(period_order) >= 2:
        st.markdown(f"### MoM Change by {selected_label}  ({prev} → {latest})")
        mom_df = dq(f"""
            SELECT "{dim_col}" AS factor,
                   COUNT(*) FILTER (WHERE _Period_label='{latest}') AS lat,
                   COUNT(*) FILTER (WHERE _Period_label='{prev}')   AS prv
            FROM tbl
            WHERE "{dim_col}" IS NOT NULL
              AND TRIM(CAST("{dim_col}" AS VARCHAR)) NOT IN ('','nan','None','N/A','-')
            GROUP BY "{dim_col}"
            HAVING (lat > 0 OR prv > 0)
        """, _summary_src)
        mom_df["delta"] = mom_df["lat"] - mom_df["prv"]
        mom_df = mom_df.sort_values("delta")
        colors_mom = [COLORS[0] if v >= 0 else COLORS[4] for v in mom_df["delta"]]
        if dmap:
            mom_df["y_label"] = mom_df["factor"].apply(
                lambda x: _short_label(x, dmap, max_chars=35))
        else:
            mom_df["y_label"] = mom_df["factor"]
        fig = go.Figure(go.Bar(
            x=mom_df["delta"],
            y=mom_df["y_label"],
            orientation="h",
            marker_color=colors_mom,
            text=mom_df["delta"].apply(lambda x: f"+{int(x)}" if x >= 0 else str(int(x))),
            textposition="outside",
            customdata=np.stack([mom_df["prv"], mom_df["lat"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Prev Period: %{customdata[0]}<br>"
                "Latest Period: %{customdata[1]}<br>"
                "Δ: %{x}<extra></extra>"
            ),
        ))
        fig.add_vline(x=0, line_dash="dash", line_color=MUTED, line_width=1)
        fig = chart_layout(fig,
            f"MoM Break Count Change by {selected_label}  ({prev} → {latest})",
            "Change in Break Count", "",
            height=max(400, len(mom_df) * 28))
        st.plotly_chart(fig, width='stretch')
        st.caption("🟩 Teal = More breaks this period (worse)   "
                   "🟥 Red = Fewer breaks this period (improvement)")

    # ─────────────────────────────────────────────────────────────────────
    # Jira Reference × System to be Fixed heatmap
    # ─────────────────────────────────────────────────────────────────────
    if (jira_ref_col  and jira_ref_col  in df.columns and
            system_col and system_col in df.columns):
        st.markdown("### Jira Reference × System to be Fixed — Break Count Heatmap")
        cross = dq(f"""
            SELECT "{jira_ref_col}" AS jira_ref,
                   "{system_col}"   AS system_fix,
                   COUNT(*)         AS cnt
            FROM tbl
            WHERE "{jira_ref_col}" IS NOT NULL AND "{system_col}" IS NOT NULL
              AND TRIM(CAST("{jira_ref_col}" AS VARCHAR)) NOT IN ('','nan','None','N/A','-')
              AND TRIM(CAST("{system_col}"   AS VARCHAR)) NOT IN ('','nan','None','N/A','-')
            GROUP BY "{jira_ref_col}", "{system_col}"
        """, df)

        if len(cross) > 0:
            if jira_desc_col and jira_desc_col in df.columns:
                desc_map_h = dq(f"""
                    SELECT "{jira_ref_col}" AS ref,
                           FIRST("{jira_desc_col}") AS d
                    FROM tbl WHERE "{jira_ref_col}" IS NOT NULL
                    GROUP BY "{jira_ref_col}"
                """, df).set_index("ref")["d"].to_dict()
                cross["jira_label"] = cross["jira_ref"].apply(
                    lambda x: _short_label(x, desc_map_h, max_chars=30))
            else:
                cross["jira_label"] = cross["jira_ref"]

            pivot = cross.pivot_table(
                index="jira_label", columns="system_fix",
                values="cnt", fill_value=0)
            jira_totals = cross.groupby("jira_label")["cnt"].sum()
            top20_labels = jira_totals.nlargest(20).index
            pivot = pivot.loc[pivot.index.isin(top20_labels)]

            fig = px.imshow(
                pivot,
                color_continuous_scale=["#E8F5E9","#FFC553","#A84B2F"],
                aspect="auto", text_auto=".0f",
            )
            fig = chart_layout(fig,
                "Jira Reference × System to be Fixed (Top 20 Jiras — Break Count)",
                "", "", height=max(400, len(pivot) * 30))
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No data found for Jira Reference × System to be Fixed cross-analysis.")


# ── Tab: FP Thresholding ──────────────────────────────────────────────────────

def tab_fp_thresholding(df_f: pd.DataFrame, col_map: dict, hist_df=None) -> None:
    st.markdown("## 🎯 Break Priority & False Positive Thresholding")

    st.markdown(
        '<div class="banner-fp">🎯 Segments are ranked by ABS GBP amount against their '
        'historical trend. High-priority = materially elevated above historical norms. '
        'Low-priority / FP candidates = within historical norms (candidates for known Jira tagging).</div>',
        unsafe_allow_html=True)

    abs_col = col_map.get("ABS GBP") or col_map.get("BREAK AMOUNT GBP")
    if not abs_col:
        st.info("No ABS GBP / Break Amount column found.")
        return

    # Combine hist + current so we have multiple periods
    _need_cols = ["_Period_label", abs_col]
    _src_frames = [df_f[[c for c in _need_cols if c in df_f.columns]]]
    if (hist_df is not None and len(hist_df) > 0
            and "_Period_label" in hist_df.columns
            and abs_col in hist_df.columns):
        _src_frames.append(hist_df[[c for c in _need_cols if c in hist_df.columns]])

    # Add segment columns from whichever frame has them
    seg_col_keys = ["Rec Name (as per Rec Cube)", "Team", "Entity", "Asset Class"]
    seg_cols = [col_map[k] for k in seg_col_keys if k in col_map and col_map[k] in df_f.columns]
    if not seg_cols:
        st.info("No segment columns (Rec Name, Team, Entity, Asset Class) found for analysis.")
        return

    # Issue Category columns (resolved early so they can ride in combined)
    issue_cat_col  = col_map.get("ISSUE CATEGORY")
    issue_cat2_col = col_map.get("ISSUE CATEGORY2")
    _extra_cols    = [c for c in [issue_cat_col, issue_cat2_col] if c and c in df_f.columns]

    # Rebuild source frames with segment cols included
    _all_cols = seg_cols + ["_Period_label", abs_col] + _extra_cols
    _src_frames = [df_f[[c for c in _all_cols if c in df_f.columns]]]
    if (hist_df is not None and len(hist_df) > 0
            and "_Period_label" in hist_df.columns
            and abs_col in hist_df.columns):
        _src_frames.append(hist_df[[c for c in _all_cols if c in hist_df.columns]])

    combined = pd.concat(_src_frames, ignore_index=True)
    if abs_col in combined.columns:
        combined[abs_col] = combined[abs_col].abs()

    period_order = sorted(combined["_Period_label"].dropna().unique().tolist())
    if len(period_order) < 2:
        st.info("Need at least 2 periods for thresholding. Upload a historical file or wait for the cache to accumulate a second period.")
        return

    latest_period = period_order[-1]
    hist_periods  = period_order[:-1]

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        seg_sel = st.selectbox(
            "Primary Segment",
            [col_map[k] for k in seg_col_keys if k in col_map and col_map[k] in combined.columns],
            format_func=lambda c: next(
                (k for k, v in col_map.items() if v == c), c),
            key="_fp_seg_sel",
            help="Group breaks by this dimension for priority ranking"
        )
    with c2:
        high_thresh = st.slider("High Priority threshold (%)", 20, 200, 50, 10,
                                help="Composite score must exceed historical mean by this % → High Priority")
    with c3:
        med_thresh = st.slider("Medium Priority threshold (%)", 5, 100, 20, 5,
                               help="Segments between medium and high thresholds → Medium Priority")
    with c4:
        cnt_weight = st.slider(
            "Break Count Weight (%)", 0, 100, 50, 10,
            key="_fp_cnt_weight",
            help="Weight given to Break Count trend vs ABS GBP trend in the composite priority score. "
                 "0 = ABS GBP only.  100 = Break Count only.  50 = equal blend."
        )

    # ── Build ABS GBP pivot: rows = segment values, cols = periods ──────────────
    grp = combined.groupby([seg_sel, "_Period_label"])[abs_col].sum().reset_index()
    pivot = grp.pivot_table(index=seg_sel, columns="_Period_label", values=abs_col, fill_value=0)
    pivot = pivot.reset_index()
    for p in period_order:
        if p not in pivot.columns:
            pivot[p] = 0.0

    hist_data   = pivot[hist_periods].values.astype(float)
    latest_data = pivot[latest_period].values.astype(float)

    hist_mean = hist_data.mean(axis=1)
    # % deviation of latest vs historical mean (signed: positive = elevated)
    dev_pct = (latest_data - hist_mean) / np.maximum(hist_mean, 1) * 100

    pivot["Hist Avg ABS GBP"] = np.round(hist_mean, 0)
    pivot["Latest ABS GBP"]   = np.round(latest_data, 0)
    pivot["vs Hist Avg %"]    = np.round(dev_pct, 1)

    # ABS GBP trend direction (slope of linear fit across historical periods)
    if len(hist_periods) >= 2:
        x = np.arange(len(hist_periods), dtype=float)
        slopes = np.array([
            np.polyfit(x, hist_data[i], 1)[0] if np.any(hist_data[i] > 0) else 0.0
            for i in range(len(hist_data))
        ])
        pivot["Trend"] = np.where(slopes > 0, "↑ Rising", np.where(slopes < 0, "↓ Falling", "→ Stable"))
    else:
        pivot["Trend"] = "—"

    # ── Break Count pivot: same structure, counting rows ─────────────────────
    grp_cnt   = combined.groupby([seg_sel, "_Period_label"]).size().reset_index(name="_cnt")
    pivot_cnt = grp_cnt.pivot_table(
        index=seg_sel, columns="_Period_label", values="_cnt", fill_value=0
    ).reset_index()
    for p in period_order:
        if p not in pivot_cnt.columns:
            pivot_cnt[p] = 0.0
    # Merge count pivot columns into main pivot (avoid column name clash)
    pivot = pivot.merge(
        pivot_cnt.rename(columns={p: f"_cnt_{p}" for p in period_order}),
        on=seg_sel, how="left",
    )

    hist_cnt_cols   = [f"_cnt_{p}" for p in hist_periods]
    hist_cnt_data   = pivot[hist_cnt_cols].values.astype(float)
    latest_cnt_data = pivot[f"_cnt_{latest_period}"].values.astype(float)
    hist_cnt_mean   = hist_cnt_data.mean(axis=1)
    dev_pct_cnt     = (latest_cnt_data - hist_cnt_mean) / np.maximum(hist_cnt_mean, 1) * 100

    pivot["Latest Break Count"]   = latest_cnt_data.astype(int)
    pivot["Hist Avg Break Count"] = np.round(hist_cnt_mean, 1)
    pivot["vs Hist Count %"]      = np.round(dev_pct_cnt, 1)

    # Break count trend direction
    if len(hist_periods) >= 2:
        cnt_slopes = np.array([
            np.polyfit(x, hist_cnt_data[i], 1)[0] if np.any(hist_cnt_data[i] > 0) else 0.0
            for i in range(len(hist_cnt_data))
        ])
        pivot["Count Trend"] = np.where(cnt_slopes > 0, "↑ Rising",
                               np.where(cnt_slopes < 0, "↓ Falling", "→ Stable"))
    else:
        pivot["Count Trend"] = "—"

    # ── Composite priority score: weighted blend of ABS GBP + Break Count ────
    w_cnt           = cnt_weight / 100.0
    composite_score = (1.0 - w_cnt) * dev_pct + w_cnt * dev_pct_cnt

    conditions = [
        composite_score >= high_thresh,
        composite_score >= med_thresh,
    ]
    pivot["Priority"] = np.select(conditions, ["🔴 High", "🟡 Medium"], default="🟢 Low / FP Candidate")
    pivot["Tag for Review"] = pivot["Priority"] == "🔴 High"

    # Per-period display columns
    for p in period_order:
        pivot[f"ABS {p}"] = pivot[p].round(0)
        pivot[f"Cnt {p}"] = pivot[f"_cnt_{p}"].fillna(0).astype(int)

    result_df = pivot.sort_values("Priority", ascending=True).reset_index(drop=True)

    # ── Summary KPIs ───────────────────────────────────────────────────────────
    n_high = int((result_df["Priority"] == "🔴 High").sum())
    n_med  = int((result_df["Priority"] == "🟡 Medium").sum())
    n_low  = int((result_df["Priority"] == "🟢 Low / FP Candidate").sum())
    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Latest Period", latest_period)
    with k2: kpi_card("High Priority", str(n_high), "segments materially elevated", warn=(n_high > 0))
    with k3: kpi_card("Medium Priority", str(n_med), "segments slightly elevated")
    with k4: kpi_card("Low / FP Candidates", str(n_low), "within historical norms")

    # ── Jira metadata enrichment: top Jira per segment value ────────────────
    jira_ref_col_fp  = col_map.get("Jira Reference")
    jira_desc_col_fp = col_map.get("Jira Desc")
    system_col_fp    = col_map.get("System to be Fixed")
    _jira_display_cols = []
    if jira_ref_col_fp and jira_ref_col_fp in df_f.columns and seg_sel in df_f.columns:
        _jira_src_cols = [seg_sel, jira_ref_col_fp]
        if jira_desc_col_fp and jira_desc_col_fp in df_f.columns:
            _jira_src_cols.append(jira_desc_col_fp)
        if system_col_fp and system_col_fp in df_f.columns:
            _jira_src_cols.append(system_col_fp)
        _grp_cols = list(dict.fromkeys(_jira_src_cols))  # deduplicate preserving order
        jira_meta = (
            df_f[_grp_cols]
            .dropna(subset=[jira_ref_col_fp])
            .groupby(_grp_cols)
            .size().reset_index(name="_cnt")
            .sort_values("_cnt", ascending=False)
            .drop_duplicates(subset=[seg_sel])
            .drop(columns="_cnt")
            .rename(columns={
                jira_ref_col_fp: "Top Jira",
                **({jira_desc_col_fp: "Jira Desc"} if jira_desc_col_fp else {}),
                **({system_col_fp:    "System to be Fixed"} if system_col_fp else {}),
            })
        )
        result_df = result_df.merge(jira_meta, on=seg_sel, how="left")
        _jira_display_cols = [c for c in ["Top Jira", "Jira Desc", "System to be Fixed"]
                              if c in result_df.columns]

    # ── Issue Category enrichment: most frequent category per segment ─────────
    _issue_display_cols = []
    if issue_cat_col and issue_cat_col in df_f.columns and seg_sel in df_f.columns:
        _issue_src = [seg_sel, issue_cat_col]
        if issue_cat2_col and issue_cat2_col in df_f.columns:
            _issue_src.append(issue_cat2_col)
        _issue_grp_cols = list(dict.fromkeys(_issue_src))  # deduplicate preserving order
        issue_meta = (
            df_f[_issue_grp_cols]
            .dropna(subset=[issue_cat_col])
            .groupby(_issue_grp_cols)
            .size().reset_index(name="_cnt")
            .sort_values("_cnt", ascending=False)
            .drop_duplicates(subset=[seg_sel])
            .drop(columns="_cnt")
            .rename(columns={
                issue_cat_col: "Issue Category",
                **({issue_cat2_col: "Issue Category 2"} if issue_cat2_col else {}),
            })
        )
        result_df = result_df.merge(issue_meta, on=seg_sel, how="left")
        _issue_display_cols = [c for c in ["Issue Category", "Issue Category 2"]
                               if c in result_df.columns]

    # ── Priority table ────────────────────────────────────────────────────────
    display_cols = (
        [seg_sel] + _jira_display_cols + _issue_display_cols +
        ["Priority",
         "Latest ABS GBP", "Hist Avg ABS GBP", "vs Hist Avg %", "Trend",
         "Latest Break Count", "Hist Avg Break Count", "vs Hist Count %", "Count Trend",
         "Tag for Review"] +
        [f"ABS {p}" for p in period_order] +
        [f"Cnt {p}" for p in period_order]
    )
    display_df = result_df[[c for c in display_cols if c in result_df.columns]]

    st.markdown(f"### Priority Ranking by {seg_sel} — Latest: {latest_period}")

    # ── Persist checkbox state across VALUE_CHANGED reruns ────────────────────
    _tag_key = f"_fp_tags_{seg_sel}"
    if _tag_key not in st.session_state:
        st.session_state[_tag_key] = {}
    display_df = display_df.copy()
    display_df["Tag for Review"] = display_df[seg_sel].astype(str).map(
        lambda x: st.session_state[_tag_key].get(x, False)
    )

    if HAS_AGGRID:
        gb_fp = GridOptionsBuilder.from_dataframe(display_df)
        gb_fp.configure_default_column(
            resizable=True, sortable=True, filter=True, floatingFilter=True
        )
        gb_fp.configure_column("Tag for Review", editable=True, width=130)
        gb_fp.configure_column(seg_sel, pinned="left", width=160)
        if "Top Jira" in display_df.columns:
            gb_fp.configure_column("Top Jira", width=140)
        if "Jira Desc" in display_df.columns:
            gb_fp.configure_column("Jira Desc", width=260)
        if "System to be Fixed" in display_df.columns:
            gb_fp.configure_column("System to be Fixed", width=200)
        if "Issue Category" in display_df.columns:
            gb_fp.configure_column("Issue Category", width=200)
        if "Issue Category 2" in display_df.columns:
            gb_fp.configure_column("Issue Category 2", width=200)
        gb_fp.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
        _apply_computed_headers(gb_fp, display_df.columns.tolist())
        fp_grid = AgGrid(
            display_df,
            gridOptions=gb_fp.build(),
            height=480,
            theme="streamlit",
            allow_unsafe_jscode=True,
            key="_fp_priority_grid",
            update_mode=GridUpdateMode.VALUE_CHANGED,
        )
        edited = pd.DataFrame(fp_grid["data"]) if fp_grid["data"] is not None else display_df
        for _, _row in edited.iterrows():
            st.session_state[_tag_key][str(_row[seg_sel])] = bool(_row.get("Tag for Review", False))
    else:
        edited = st.data_editor(
            display_df,
            width='stretch',
            hide_index=True,
            column_config={
                "Tag for Review":      st.column_config.CheckboxColumn("Tag for Review", default=False),
                "Latest ABS GBP":      st.column_config.NumberColumn(format="£%.0f"),
                "Hist Avg ABS GBP":    st.column_config.NumberColumn(format="£%.0f"),
                "vs Hist Avg %":       st.column_config.NumberColumn(format="%.1f%%"),
                "Latest Break Count":  st.column_config.NumberColumn(format="%d"),
                "Hist Avg Break Count":st.column_config.NumberColumn(format="%.1f"),
                "vs Hist Count %":     st.column_config.NumberColumn(format="%.1f%%"),
            },
            key="_fp_editor",
        )

    if st.button("Apply Tagged Segments to Filters", key="_fp_apply"):
        confirmed = edited[edited["Tag for Review"] == True]
        if len(confirmed) > 0:
            fp_seg_keys = list(confirmed[seg_sel].astype(str).tolist())
            st.session_state["_fp_seg_keys_v2"] = fp_seg_keys
            st.session_state["_fp_seg_col_v2"]  = seg_sel
            st.success(f"✅ {len(fp_seg_keys)} segments tagged for review.")
        else:
            st.info("No segments tagged.")

    # ── ABS GBP Trend chart for top-10 segments ────────────────────────────────
    top10_segs = result_df.nlargest(10, "Latest ABS GBP")[seg_sel].tolist()
    trend_rows = grp[grp[seg_sel].isin(top10_segs)].copy()
    trend_rows[abs_col] = trend_rows[abs_col].round(0)
    if len(trend_rows) > 0:
        fig_t = px.line(
            trend_rows.sort_values("_Period_label"),
            x="_Period_label", y=abs_col, color=seg_sel,
            color_discrete_sequence=COLORS, markers=True,
        )
        fig_t = chart_layout(fig_t, f"ABS GBP Trend — Top 10 by {seg_sel}",
                             "Period", "ABS GBP (£)", height=420)
        st.plotly_chart(fig_t, width='stretch')

    # ── Break Count Trend chart for top-10 segments by count ──────────────────
    top10_by_cnt = result_df.nlargest(10, "Latest Break Count")[seg_sel].tolist()
    cnt_trend_rows = grp_cnt[grp_cnt[seg_sel].isin(top10_by_cnt)].copy()
    if len(cnt_trend_rows) > 0:
        fig_c = px.line(
            cnt_trend_rows.sort_values("_Period_label"),
            x="_Period_label", y="_cnt", color=seg_sel,
            color_discrete_sequence=COLORS, markers=True,
        )
        fig_c = chart_layout(fig_c, f"Break Count Trend — Top 10 by {seg_sel}",
                             "Period", "Break Count", height=420)
        st.plotly_chart(fig_c, width='stretch')

    # Download
    st.download_button(
        "📥 Download Priority Analysis (CSV)",
        display_df.to_csv(index=False).encode("utf-8"),
        file_name="break_priority_thresholding.csv",
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
        st.plotly_chart(fig, width='stretch')

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
                st.plotly_chart(fig, width='stretch')
    with w2:
        if team_col:
            fig = make_waterfall_chart(team_col, "Break Count Delta by Team", "count")
            if fig:
                st.plotly_chart(fig, width='stretch')

    if abs_col and abs_col in df_f.columns:
        w3, w4 = st.columns(2)
        with w3:
            if rec_col:
                fig = make_waterfall_chart(rec_col, "ABS GBP Delta by Rec Name", "amount")
                if fig:
                    st.plotly_chart(fig, width='stretch')
        with w4:
            if team_col:
                fig = make_waterfall_chart(team_col, "ABS GBP Delta by Team", "amount")
                if fig:
                    st.plotly_chart(fig, width='stretch')

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
    st.sidebar.caption(
        "Upload the current period file (required). Optionally upload a historical "
        "file with previous periods to enable trend comparison."
    )

    # ── Cache Management ──
    with st.sidebar.expander("🗑️ Cache Management", expanded=False):
        cache_files = [f for f in os.listdir(_cache_dir()) if f.endswith(".parquet")]
        st.caption(f"Cached files: {len(cache_files)}")
        if st.button("Clear Cache & Reset Session", key="_clear_cache_btn"):
            n = clear_all_cache()
            st.success(f"Cleared {n} cached file(s). Upload a new file to begin.")
            st.rerun()

    # ── Jira Integration ──
    with st.sidebar.expander("🔗 Jira Integration", expanded=False):
        if not HAS_JIRA:
            st.warning("Install the `jira` package to enable live Jira enrichment.\n\n`pip install jira`")
        else:
            st.caption("Credentials are used only this session and never stored to disk.")
            _jira_url_inp   = st.text_input(
                "Jira URL", value=st.session_state.get("_jira_url", ""),
                placeholder="https://yourcompany.atlassian.net", key="_jira_url_inp")
            _jira_email_inp = st.text_input(
                "Email", value=st.session_state.get("_jira_email", ""),
                key="_jira_email_inp")
            _jira_token_inp = st.text_input(
                "Password / API Token", value=st.session_state.get("_jira_token", ""),
                type="password", key="_jira_token_inp")
            if st.button("Save Credentials", key="_jira_save"):
                st.session_state["_jira_url"]   = _jira_url_inp
                st.session_state["_jira_email"] = _jira_email_inp
                st.session_state["_jira_token"] = _jira_token_inp
                # Clear all Jira metadata so next tab load re-fetches with new creds
                for _k in [k for k in list(st.session_state.keys())
                           if k.startswith("_jira_meta_") or k == "_jira_cred_hash"]:
                    del st.session_state[_k]
                st.success("Credentials saved. Metadata will refresh on next tab load.")
            if st.button("🗑️ Clear Jira Metadata", key="_jira_clear_meta",
                         help="Remove all cached Jira API results so they are re-fetched"):
                _cleared = 0
                for _k in [k for k in list(st.session_state.keys())
                           if k in ("_jira_meta_store", "_jira_cred_hash")
                           or k.startswith("_jira_meta_")]:
                    del st.session_state[_k]
                    _cleared += 1
                st.success(f"Jira metadata cache cleared ({_cleared} key(s) removed).")
            if st.session_state.get("_jira_url"):
                st.caption(f"Connected to: {st.session_state['_jira_url']}")

    uploaded_latest = st.sidebar.file_uploader(
        "📂 Latest Period File",
        type=["xlsx", "xls", "csv", "parquet"],
        key="_file_latest",
        help="Current period's reconciliation break data (Excel, CSV, or Parquet).",
    )

    uploaded_hist = st.sidebar.file_uploader(
        "📂 Historical Data (optional)",
        type=["xlsx", "xls", "csv", "parquet"],
        key="_file_hist",
        help="Previous periods file for historical comparison. Shown as expanders in Break Counts, Amount Analysis, and Jira Factor Analysis tabs.",
    )

    if st.sidebar.button("Reset Filters", key="_reset_btn"):
        _reset_filters()
        st.rerun()

    if uploaded_latest is None:
        st.markdown(
            '<div class="banner-info">👆 Upload the <b>Latest Period File</b> in the sidebar to begin analysis.</div>',
            unsafe_allow_html=True)
        st.stop()

    with st.spinner("Processing latest period file..."):
        raw_bytes = uploaded_latest.read()
        result = run_pipeline(raw_bytes)

    df      = result["df"]
    col_map = result["col_map"]

    # Set overflow in session state
    st.session_state["_overflow"] = result["overflow_count"]

    if result.get("cached"):
        st.sidebar.success("⚡ Loaded from cache.")

    # Parquet download button — lets users save the converted file for fast re-upload
    pq_path = _pq_path(result["fhash"])
    if os.path.exists(pq_path):
        with open(pq_path, "rb") as _f:
            _pq_bytes = _f.read()
        st.sidebar.download_button(
            label="⬇️ Download as Parquet",
            data=_pq_bytes,
            file_name=f"breaks_{result['fhash'][:8]}.parquet",
            mime="application/octet-stream",
            help="Download the cached Parquet file for instant re-upload next time.",
        )

    # Historical data: explicit upload takes priority over auto-cached periods
    if uploaded_hist is not None:
        with st.spinner("Processing historical file..."):
            hist_bytes = uploaded_hist.read()
            hist_result = run_pipeline(hist_bytes)
        hist_df = hist_result["df"]
    else:
        current_periods = df["_Period_label"].dropna().unique().tolist() if "_Period_label" in df.columns else []
        hist_df = load_historical_context(current_periods, result["fhash"])

    # Build filters and apply
    filters = build_sidebar_filters(df, col_map)
    df_f = apply_filters(df, filters)

    # Inject CSS for computed-column yellow headers in all AgGrid tables
    st.markdown("""
<style>
.computed-col-header {
    background-color: #FFF176 !important;
    color: #212121 !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🧹 Data Quality & Ageing",
        "🔍 Jira Factor Analysis",
        "🎯 FP Thresholding",
    ])

    with tab1:
        tab_quality_and_ageing(df, df_f, col_map, hist_df)
    with tab2:
        tab_jira_factor_analysis(df_f, col_map, hist_df)
    with tab3:
        tab_fp_thresholding(df_f, col_map, hist_df)


if __name__ == "__main__":
    main()
