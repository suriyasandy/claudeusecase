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
    "Age Days": ["age days", "age_days", "days aged", "days old"],
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

def tab_break_counts(df_f: pd.DataFrame, col_map: dict, hist_df=None) -> None:
    st.markdown("## 📈 Break Counts + Drill-Down")

    rec_actual  = col_map.get("Rec Name (as per Rec Cube)")
    team_actual = col_map.get("Team")

    if "_Period_label" not in df_f.columns:
        st.info("No period data available.")
        return

    # Top-10 Rec Names trend
    if rec_actual and rec_actual in df_f.columns:
        # Combine hist + current so trend spans all available periods
        _trend_cols = [rec_actual, "_Period_label"]
        if (hist_df is not None and len(hist_df) > 0
                and "_Period_label" in hist_df.columns
                and rec_actual in hist_df.columns):
            _trend_src = pd.concat(
                [hist_df[_trend_cols], df_f[_trend_cols]], ignore_index=True
            )
        else:
            _trend_src = df_f[_trend_cols]
        top10_series = _trend_src[rec_actual].value_counts().head(10).index.tolist()
        top10_df = _trend_src[_trend_src[rec_actual].isin(top10_series)]
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

                    # ── Toggle controls ───────────────────────────────────────
                    entity_actual = col_map.get("Entity")
                    type_actual   = col_map.get("Type of Break")
                    ac_actual     = col_map.get("Asset Class")
                    abs_actual    = col_map.get("ABS GBP") or col_map.get("BREAK AMOUNT GBP")

                    # Build available breakdown dims (only those present in data)
                    _dim_opts = []
                    if team_actual   and team_actual   in rec_df.columns: _dim_opts.append("Team")
                    if entity_actual and entity_actual in rec_df.columns: _dim_opts.append("Entity")
                    if type_actual   and type_actual   in rec_df.columns: _dim_opts.append("Type of Break")
                    if ac_actual     and ac_actual     in rec_df.columns: _dim_opts.append("Asset Class")

                    _has_amount = bool(abs_actual and abs_actual in rec_df.columns)
                    _metric_opts = ["Count", "Amount"] if _has_amount else ["Count"]

                    tc1, tc2 = st.columns([2, 3])
                    with tc1:
                        _drill_metric = st.radio(
                            "Metric", _metric_opts, horizontal=True,
                            key="_drill_metric",
                        )
                    with tc2:
                        if _dim_opts:
                            _drill_dim = st.selectbox(
                                "Breakdown by", _dim_opts, key="_drill_dim",
                            )
                        else:
                            _drill_dim = None

                    # ── Single chart based on toggle selection ─────────────────
                    if _drill_metric == "Count" and _drill_dim:
                        _dim_col = {
                            "Team":          team_actual,
                            "Entity":        entity_actual,
                            "Type of Break": type_actual,
                            "Asset Class":   ac_actual,
                        }[_drill_dim]
                        period_cnt = dq_local(
                            f'SELECT "_Period_label", "{_dim_col}", COUNT(*) AS cnt '
                            f'FROM rec_tbl GROUP BY "_Period_label", "{_dim_col}" '
                            f'ORDER BY "_Period_label"',
                            rec_tbl=rec_df
                        )
                        fig_d = px.bar(
                            period_cnt, x="_Period_label", y="cnt", color=_dim_col,
                            color_discrete_sequence=COLORS, barmode="stack",
                        )
                        fig_d = chart_layout(
                            fig_d,
                            f"Breaks by Period × {_drill_dim} — {selected_rec}",
                            "Period", "Count", height=380,
                        )
                        st.plotly_chart(fig_d, use_container_width=True)

                    elif _drill_metric == "Count" and not _drill_dim:
                        # Fallback: plain period trend
                        period_cnt = dq_local(
                            'SELECT "_Period_label", COUNT(*) AS cnt '
                            'FROM rec_tbl GROUP BY "_Period_label" ORDER BY "_Period_label"',
                            rec_tbl=rec_df
                        )
                        fig_d = px.bar(period_cnt, x="_Period_label", y="cnt",
                                       color_discrete_sequence=[PRIMARY])
                        fig_d = chart_layout(fig_d, f"Breaks by Period — {selected_rec}",
                                             "Period", "Count", height=360)
                        st.plotly_chart(fig_d, use_container_width=True)

                    elif _drill_metric == "Amount":
                        if _drill_dim and _drill_dim in ("Team", "Entity"):
                            _dim_col = team_actual if _drill_dim == "Team" else entity_actual
                            period_amt_stk = dq_local(
                                f'SELECT "_Period_label", "{_dim_col}", '
                                f'SUM({safe_amt(abs_actual)}) AS total_amt '
                                f'FROM rec_tbl GROUP BY "_Period_label", "{_dim_col}" '
                                f'ORDER BY "_Period_label"',
                                rec_tbl=rec_df
                            )
                            fig_d = px.bar(
                                period_amt_stk, x="_Period_label", y="total_amt",
                                color=_dim_col, color_discrete_sequence=COLORS, barmode="stack",
                            )
                            fig_d = chart_layout(
                                fig_d,
                                f"ABS GBP by Period × {_drill_dim} — {selected_rec}",
                                "Period", "ABS GBP (£)", height=380,
                            )
                        else:
                            # Amount by period only (no useful stack for Type/Asset Class)
                            period_amt = dq_local(
                                f'SELECT "_Period_label", SUM({safe_amt(abs_actual)}) AS total_amt '
                                f'FROM rec_tbl GROUP BY "_Period_label" ORDER BY "_Period_label"',
                                rec_tbl=rec_df
                            )
                            fig_d = px.bar(period_amt, x="_Period_label", y="total_amt",
                                           color_discrete_sequence=[PRIMARY])
                            fig_d.update_traces(
                                text=[format_short(v) for v in period_amt["total_amt"]],
                                textposition="outside",
                            )
                            fig_d = chart_layout(
                                fig_d, f"ABS GBP by Period — {selected_rec}",
                                "Period", "ABS GBP (£)", height=360,
                            )
                        st.plotly_chart(fig_d, use_container_width=True)

                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        # No rec column — just show overall period trend
        if "_Period_label" in df_f.columns:
            period_cnt = df_f.groupby("_Period_label").size().reset_index(name="Count").sort_values("_Period_label")
            fig = px.bar(period_cnt, x="_Period_label", y="Count", color_discrete_sequence=[PRIMARY])
            fig = chart_layout(fig, "Break Count by Period", "Period", "Count", height=380)
            st.plotly_chart(fig, use_container_width=True)


# ── Tab: Amount Analysis ──────────────────────────────────────────────────────

def tab_amount_analysis(df_f: pd.DataFrame, col_map: dict, hist_df=None) -> None:
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

    # ── Chart view toggle ─────────────────────────────────────────────────────
    rec_actual  = col_map.get("Rec Name (as per Rec Cube)")
    team_actual = col_map.get("Team")

    _view_opts = ["By Period"]
    if rec_actual  and rec_actual  in df_f.columns: _view_opts.append("By Rec Name")
    if team_actual and team_actual in df_f.columns: _view_opts.append("By Team")
    _view_opts.append("Distribution")

    _amt_view = st.radio("View", _view_opts, horizontal=True, key="_amt_view")

    if _amt_view == "By Period" and "_Period_label" in df_f.columns:
        # Merge historical periods so the chart shows full trend
        if (hist_df is not None and len(hist_df) > 0
                and "_Period_label" in hist_df.columns
                and amt_col in hist_df.columns):
            _period_src = pd.concat(
                [hist_df[["_Period_label", amt_col]],
                 df_f[["_Period_label", amt_col]]], ignore_index=True
            )
        else:
            _period_src = df_f[["_Period_label", amt_col]]
        period_amt = dq_local(
            f'SELECT "_Period_label", SUM({safe_amt(amt_col)}) AS total_amt '
            f'FROM tbl GROUP BY "_Period_label" ORDER BY "_Period_label"',
            tbl=_period_src
        )
        fig = px.bar(
            period_amt, x="_Period_label", y="total_amt",
            color_discrete_sequence=[PRIMARY],
            text=[format_short(v) for v in period_amt["total_amt"]],
        )
        fig.update_traces(textposition="outside")
        fig = chart_layout(fig, "Total ABS Amount (£) by Period", "Period", "ABS GBP (£)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    elif _amt_view == "By Rec Name" and rec_actual and rec_actual in df_f.columns:
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
        fig2 = chart_layout(fig2, "Top-10 Rec Names by ABS Amount (£)", rec_actual, "ABS GBP (£)", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    elif _amt_view == "By Team" and team_actual and team_actual in df_f.columns:
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
        fig3 = chart_layout(fig3, "ABS Amount (£) by Team", team_actual, "ABS GBP (£)", height=400)
        st.plotly_chart(fig3, use_container_width=True)

    else:  # Distribution
        amt_data = df_f[amt_col].dropna()
        fig4 = px.histogram(amt_data, nbins=50, color_discrete_sequence=[PRIMARY])
        fig4 = chart_layout(fig4, "Distribution of ABS Amounts (£)", "ABS Amount (£)", "Count", height=400)
        st.plotly_chart(fig4, use_container_width=True)


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
    period_expr = (
        f'COUNT(*) FILTER (WHERE _Period_label = \'{latest}\') AS "Latest Period",'
        f'COUNT(*) FILTER (WHERE _Period_label = \'{prev}\')   AS "Prev Period"'
        if period_order else
        '0 AS "Latest Period", 0 AS "Prev Period"'
    )

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

    if "Latest Period" in summary_df.columns and "Prev Period" in summary_df.columns:
        summary_df["MoM Δ"]   = summary_df["Latest Period"] - summary_df["Prev Period"]
        summary_df["MoM Δ %"] = (
            (summary_df["MoM Δ"] /
             summary_df["Prev Period"].replace(0, np.nan)) * 100
        ).round(1)

    # ── Column order — Jira Desc always immediately after Jira Reference ──
    col_order = [selected_label]
    for c in ["Jira Description", "System to be Fixed", "Account Group",
              "Products Reconciled", "Unique Jira Refs"]:
        if c in summary_df.columns: col_order.append(c)
    for c in ["Break Count", "Latest Period", "Prev Period", "MoM Δ", "MoM Δ %",
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
        gb.configure_default_column(filterable=True, sortable=True, resizable=True,
                                    wrapText=True, autoHeight=True)
        gb.configure_column(selected_label, pinned="left", width=160)
        if "Jira Description" in summary_df.columns:
            gb.configure_column("Jira Description", width=300,
                                tooltipField="Jira Description")
        if "System to be Fixed" in summary_df.columns:
            gb.configure_column("System to be Fixed", width=200)
        for c in ["Break Count", "Latest Period", "Prev Period"]:
            if c in summary_df.columns:
                gb.configure_column(c, width=120)
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
        AgGrid(summary_df, gridOptions=gb.build(), height=440,
               theme="streamlit", allow_unsafe_jscode=True, key="jira_summary_grid")
    else:
        st.dataframe(summary_df, height=440, use_container_width=True)

    # ── Download button right below the table ──
    st.download_button(
        f"📥 Download {selected_label} Factor Summary (CSV)",
        summary_df.to_csv(index=False).encode("utf-8"),
        file_name=f"factor_{selected_label.replace(' ','_').lower()}.csv",
        mime="text/csv",
    )

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
        st.plotly_chart(fig, use_container_width=True)

    # ── Chart 2: Top 15 by Avg Age Days (colour-coded by severity) ──
    with row1b:
        if "_Computed_Age_Days" in df.columns:
            top_age = dq(f"""
                SELECT "{dim_col}" AS factor,
                       ROUND(AVG(_Computed_Age_Days), 1) AS avg_age
                FROM tbl
                WHERE "{dim_col}" IS NOT NULL
                  AND TRIM(CAST("{dim_col}" AS VARCHAR)) NOT IN ('','nan','None','N/A','-')
                  AND _Computed_Age_Days IS NOT NULL
                GROUP BY "{dim_col}" ORDER BY avg_age DESC LIMIT 15
            """, df).sort_values("avg_age", ascending=True)
            bar_colors = [
                WARN      if v > 180 else
                COLORS[8] if v > 90  else
                COLORS[0]
                for v in top_age["avg_age"]
            ]
            if dmap:
                top_age["hover_desc"] = top_age["factor"].map(dmap).fillna("")
                htemplate = "<b>%{y}</b><br>%{customdata}<br>Avg Age: %{x}d<extra></extra>"
                fig = go.Figure(go.Bar(
                    x=top_age["avg_age"], y=top_age["factor"],
                    orientation="h", marker_color=bar_colors,
                    text=top_age["avg_age"].astype(str) + "d",
                    textposition="outside",
                    customdata=top_age["hover_desc"],
                    hovertemplate=htemplate,
                ))
            else:
                fig = go.Figure(go.Bar(
                    x=top_age["avg_age"], y=top_age["factor"],
                    orientation="h", marker_color=bar_colors,
                    text=top_age["avg_age"].astype(str) + "d",
                    textposition="outside",
                ))
            fig.add_vline(x=90,  line_dash="dash", line_color=COLORS[8], line_width=1.2,
                          annotation_text="90d",  annotation_position="top right")
            fig.add_vline(x=180, line_dash="dash", line_color=WARN,      line_width=1.2,
                          annotation_text="180d", annotation_position="top right")
            fig = chart_layout(fig, f"Top 15 {selected_label} by Avg Age Days",
                               "Avg Age Days", "", height=max(360, len(top_age) * 32))
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

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
                st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # Trend over periods — Top 5, legend annotated with Jira Desc
    # ─────────────────────────────────────────────────────────────────────
    if "_Period_label" in df.columns and period_order:
        st.markdown(f"### {selected_label} — Break Count Trend Over Periods (Top 5)")
        top5 = dq(f"""
            SELECT "{dim_col}" AS factor FROM tbl
            WHERE "{dim_col}" IS NOT NULL
              AND TRIM(CAST("{dim_col}" AS VARCHAR)) NOT IN ('','nan','None','N/A','-')
            GROUP BY "{dim_col}" ORDER BY COUNT(*) DESC LIMIT 5
        """, df)["factor"].tolist()
        top5_str = ", ".join(f"'{str(v).replace(chr(39), chr(39)*2)}'" for v in top5)
        trend_df = dq(f"""
            SELECT "{dim_col}" AS factor, _Period_label AS period, COUNT(*) AS cnt
            FROM tbl WHERE "{dim_col}" IN ({top5_str})
            GROUP BY "{dim_col}", _Period_label ORDER BY _Period_label
        """, df)
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
        st.plotly_chart(fig, use_container_width=True)

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
        """, df)
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
        st.plotly_chart(fig, use_container_width=True)
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
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data found for Jira Reference × System to be Fixed cross-analysis.")


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
    st.sidebar.caption(
        "Upload the current period file (required). Optionally upload a historical "
        "file with previous periods to enable trend comparison."
    )

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

    # Tabs — Period Comparison removed; historical data surfaced inside each tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧹 Data Quality",
        "📅 Ageing Validation",
        "📈 Break Counts + Drill-Down",
        "💷 Amount Analysis",
        "🔍 Jira Factor Analysis",
        "🎯 FP Thresholding",
    ])

    with tab1:
        tab_data_quality(df, df_f, col_map)
    with tab2:
        tab_ageing_validation(df_f, col_map)
    with tab3:
        tab_break_counts(df_f, col_map, hist_df)
    with tab4:
        tab_amount_analysis(df_f, col_map, hist_df)
    with tab5:
        tab_jira_factor_analysis(df_f, col_map, hist_df)
    with tab6:
        tab_fp_thresholding(df_f, col_map)


if __name__ == "__main__":
    main()
