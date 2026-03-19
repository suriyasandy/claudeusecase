# Accounting Control Breaks — AI Analysis Platform

A Streamlit application for analysing, tracking, and prioritising accounting control breaks.
Upload your breaks data and instantly get trend charts, factor analysis, ageing breakdowns,
and priority ranking across periods — with automatic historical caching so every upload enriches
the trend views.

---

## Getting Started

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed in the terminal (default: `http://localhost:8501`).

---

## Uploading Data

### Primary File (required)
Upload a CSV, Excel (.xlsx), or Parquet file containing your control breaks.
The app auto-detects column roles using fuzzy matching — standard column names like
`Rec Name`, `Team`, `Entity`, `Period`, `Age Days`, `ABS GBP`, `Jira Reference`, etc. are
recognised automatically.

### Historical File (optional)
Upload a second file from a previous period if you want to manually supply historical data.
If left empty, the app uses its **automatic cache** — every file you upload is cached locally,
so on subsequent uploads the previous periods are available automatically without re-uploading.

---

## Tabs

### Data Quality
Shows column mapping, missing-value coverage, duplicate detection, and a preview of raw data.
Use this tab to verify that the app has correctly identified your columns before analysing.

### Ageing Validation
Breaks down open items by age bucket (0–15 days, 16–30 days, …, 365+ days).
A configurable "as-of date" lets you recompute ages for any historical point in time.
Includes KPI cards (total breaks, >90-day count, average age) and an Avg Age Days trend
chart spanning all available periods including cached history.

### Break Counts + Drill-Down
Shows the top-10 Rec Names as a period trend line chart using all available historical
and current periods.

**Drill-Down**: Select any Rec Name from the dropdown to drill into it.
Choose a **Metric** (Count or ABS GBP Amount) and a **Breakdown** dimension (Team, Entity,
Type of Break, Asset Class). The stacked bar chart shows all periods — historical and current —
so you can see how that rec's break profile has evolved over time.

### Amount Analysis
Analyses ABS GBP break amounts. The **By Period** view merges historical and current data to
show a full timeline bar chart. Switch to **By Rec Name**, **By Team**, or **Distribution**
to view the current period in detail.

### Jira Factor Analysis
Groups breaks by a chosen dimension — Jira Reference, System to be Fixed, Rec Name, Team,
Entity, Asset Class, or True/Systemic indicator — and builds a summary table.

Each row shows break counts for **every available period** (one column per period, oldest to
newest), plus **MoM Δ** and **MoM Δ %** comparing the two most recent periods, ageing metrics,
and ABS GBP totals. Sort any column without triggering a page reload.

### FP Thresholding (Break Priority Ranking)
Ranks each segment (Rec Name / Team / Entity / Asset Class — your choice) by its **ABS GBP
amount** relative to its own historical baseline:

| Priority | Meaning |
|---|---|
| 🔴 High | Latest ABS GBP materially exceeds historical mean — needs attention |
| 🟡 Medium | Slightly elevated — monitor |
| 🟢 Low / FP Candidate | Within historical norms — likely a known / recurring break |

The table also shows trend direction (↑ Rising / ↓ Falling / → Stable) and per-period ABS GBP
columns. Tick the **Tag for Review** checkbox on any rows and click **Apply Tagged Segments to
Filters** to register them. Then enable **Exclude Confirmed False Positives** in the sidebar
and click **Apply Filters** to remove those segments from all other tabs — useful for isolating
genuinely new or anomalous breaks.

---

## Historical Data & Caching

Every time you upload a file, the app automatically saves it to a local Parquet cache
(in your system's temp directory). On the next upload, previous periods that are not in
the current file are loaded from this cache and merged into all trend charts, the Factor
Analysis summary table, and the FP Thresholding tab — no re-upload needed.

The cache is keyed by file content hash, so uploading the same file twice does not create
duplicate entries. You can also download the cached Parquet from the sidebar for faster
future uploads.

---

## Filters

The left sidebar provides:

- **Period** — multi-select to focus on specific months
- **Team / Entity / Rec Name / Type of Break / Asset Class** — narrow by any dimension
- **Apply Filters** — filters are not applied live; click this button to update all charts
- **Reset Filters** — restore all filters to their defaults
- **Exclude Confirmed False Positives** — when enabled, removes segments tagged in the
  FP Thresholding tab from all charts (see above)

A warning banner appears whenever you change a filter widget without applying it, so you always
know whether the charts reflect your current selections.
