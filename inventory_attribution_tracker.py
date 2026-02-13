import os
import json
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# CONFIG (YOUR PATHS)
# =========================================================
BASE_DIR = Path(r"C:\00 VIP\01 Python\market_analysis\inventory_attribution_analysis")
DATA_DIR = BASE_DIR / "data"

OUT_CSV = DATA_DIR / "inventory_attribution_weekly.csv"
OUT_HTML = BASE_DIR / "index.html"

KEEP_LAST_WEEKS = 520  # ~10 years

# Robust attribution parameters
FLOW_CHANGE_THRESHOLD_MB_W = 1.0  # below this, consider changes minor/noise
CONFIDENCE_RATIO = 1.35           # top driver must be >= 1.35x second for "High"
ZSCORE_WINDOW = 52                # rolling window (~1 year)

SERIES = {
    "inventory_kb": "PET.WCESTUS1.W",  # weekly inventory level, thousand barrels (kb)
    "imports_kbd": "PET.WCRIMUS2.W",   # crude imports, kb/d
    "exports_kbd": "PET.WCREXUS2.W",   # crude exports, kb/d
    "runs_kbd": "PET.WCRRIUS2.W",      # refinery crude inputs, kb/d
}

# =========================================================
# HELPERS
# =========================================================
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_api_key() -> str:
    load_dotenv()
    key = os.getenv("EIA_API_KEY")
    if not key:
        raise ValueError("EIA_API_KEY not found in .env")
    return key

def fetch_eia_series(series_id: str, api_key: str) -> pd.DataFrame:
    url = f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={api_key}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["period"])
    return df[["date", "value"]].sort_values("date")

def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - mu) / sd

def fmt_signed(x: float, dp: int = 2) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{dp}f}"

def pick_driver(row) -> tuple[str, str]:
    """
    Robust dominant driver:
    - compare ABS WoW CHANGES of flows (mb/week), not levels
    - apply materiality threshold
    - assign confidence by top-vs-second ratio
    """
    changes = {
        "Import-led": abs(row["imports_change_mb_w"]),
        "Export-led": abs(row["exports_change_mb_w"]),
        "Run-led": abs(row["runs_change_mb_w"]),
    }
    ranked = sorted(changes.items(), key=lambda kv: kv[1], reverse=True)
    top_driver, top_val = ranked[0]
    _, second_val = ranked[1]

    if top_val < FLOW_CHANGE_THRESHOLD_MB_W:
        return ("Mixed / Minor", "Low")

    ratio = (top_val / second_val) if second_val > 0 else float("inf")
    if ratio >= CONFIDENCE_RATIO:
        return (top_driver, "High")
    return (top_driver, "Medium")

def most_abnormal_flow(row) -> str:
    z = {
        "Imports": abs(row["imports_change_z"]) if pd.notna(row["imports_change_z"]) else 0,
        "Exports": abs(row["exports_change_z"]) if pd.notna(row["exports_change_z"]) else 0,
        "Runs": abs(row["runs_change_z"]) if pd.notna(row["runs_change_z"]) else 0,
    }
    return max(z, key=z.get)

def _tier_inventory_signal(inv_abs_mb: float) -> str:
    # Inventory magnitude sets how much weight we give a single print
    if inv_abs_mb >= 3.0:
        return "High"
    if inv_abs_mb >= 1.0:
        return "Medium"
    return "Low"

def _offset_flag(imports_chg: float, exports_chg: float, diff_mb: float = 0.50, ratio: float = 0.20) -> tuple[bool, str]:
    """
    Detect near-offsetting two-way trade (imports and exports move together).
    - absolute diff threshold (mb/week)
    - relative diff threshold vs the larger leg
    """
    a = abs(imports_chg)
    b = abs(exports_chg)
    denom = max(a, b, 1e-9)
    rel = abs(imports_chg - exports_chg) / denom

    is_offset = (abs(imports_chg - exports_chg) <= diff_mb) or (rel <= ratio and max(a, b) >= FLOW_CHANGE_THRESHOLD_MB_W)
    reason = f"Imports/exports moved in tandem (Δ={abs(imports_chg-exports_chg):.2f} mb/w; rel={rel:.0%})." if is_offset else ""
    return is_offset, reason

def _conviction(inventory_signal: str, driver_conf: str, is_offset: bool) -> str:
    """
    Conviction combines:
    - inventory magnitude (signal strength)
    - driver confidence (separation)
    - offsetting flows (reduces directional clarity)
    """
    score = 0
    score += {"Low": 0, "Medium": 1, "High": 2}[inventory_signal]
    score += {"Low": 0, "Medium": 1, "High": 2}[driver_conf]
    if is_offset:
        score -= 1  # offsetting trade flows reduce directional conviction

    if score >= 3:
        return "High"
    if score == 2:
        return "Medium"
    return "Low"

def _bias(inv_chg: float, conviction: str) -> str:
    """
    Bias is inventory-direction informed, but only meaningful if conviction isn't Low.
    """
    if conviction == "Low":
        return "Neutral (low conviction)"
    if inv_chg <= -1.0:
        return "Tightening bias (supports prompt strength) if persistent"
    if inv_chg >= 1.0:
        return "Loosening bias (pressures prompt strength) if persistent"
    return "Neutral"

def _abnormality_sentence(row: pd.Series) -> str:
    """
    Use z-scores to add statistical weight.
    """
    flow = row["most_abnormal_flow"]  # "Imports" / "Exports" / "Runs"
    z_map = {
        "Imports": row.get("imports_change_z", 0.0),
        "Exports": row.get("exports_change_z", 0.0),
        "Runs": row.get("runs_change_z", 0.0),
    }
    z = z_map.get(flow, 0.0)
    if pd.isna(z):
        return "Statistical context: insufficient history to score abnormality."
    if abs(z) >= 2.0:
        return f"Statistical context: {flow} move is statistically unusual (z={z:+.2f})."
    return f"Statistical context: {flow} is the most abnormal mover, but within typical variance (z={z:+.2f})."

def build_refined_commentary(row: pd.Series) -> str:
    """
    Senior-style commentary that:
    - handles near-offsetting flows
    - avoids overcalling dominance
    - clearly separates mechanics vs bias vs conviction
    """
    d = row["date_str"]
    inv_chg = float(row["inventory_change_mb"])
    inv_abs = abs(inv_chg)
    inv_word = "draw" if inv_chg < 0 else "build" if inv_chg > 0 else "flat"

    imp = float(row["imports_change_mb_w"])
    exp = float(row["exports_change_mb_w"])
    run = float(row["runs_change_mb_w"])

    net_external = imp - exp  # +ve net inflow; -ve net outflow
    driver = row["dominant_driver"]
    driver_conf = row["driver_confidence"]

    # Detect near-offsetting imports/exports (two-way trade)
    is_offset, offset_reason = _offset_flag(imp, exp)

    # Inventory magnitude signal
    inv_signal = _tier_inventory_signal(inv_abs)

    # Conviction (integrated)
    conv = _conviction(inv_signal, driver_conf, is_offset)

    # Bias (only meaningful if conviction not Low)
    bias = _bias(inv_chg, conv)

    # Mechanics framing (avoid overclaim)
    flow_config = []
    if abs(net_external) <= 0.50 and max(abs(imp), abs(exp)) >= FLOW_CHANGE_THRESHOLD_MB_W:
        flow_config.append("Two-way trade flows broadly offset (imports and exports moved together).")
    elif net_external > 0.50:
        flow_config.append("Net external balance shifted toward inflow (imports outpaced exports).")
    elif net_external < -0.50:
        flow_config.append("Net external balance shifted toward outflow (exports outpaced imports).")
    else:
        flow_config.append("Net external shift is modest; directional signal limited.")

    if run >= 1.0:
        flow_config.append("Refinery demand rose meaningfully (runs up).")
    elif run <= -1.0:
        flow_config.append("Refinery demand softened meaningfully (runs down).")
    else:
        flow_config.append("Refinery runs changed modestly.")

    # Driver phrasing: downgrade if offsetting
    if is_offset and driver in ("Import-led", "Export-led"):
        driver_line = f"Driver framing: offsetting imports/exports reduce dominance clarity (flagged as {driver}, but treated as two-way flow week)."
    else:
        driver_line = f"Driver call: {driver} (confidence: {driver_conf})."

    # Conclusion: match print + configuration
    if inv_word == "build":
        if is_offset:
            conclusion = "Conclusion: The build appears flow-intensity/timing-driven rather than a clear structural loosening signal."
        elif net_external > 0.50 and run < 1.0:
            conclusion = "Conclusion: The build is consistent with net inflow not fully absorbed by refinery demand."
        elif run <= -1.0:
            conclusion = "Conclusion: The build is consistent with weaker refinery demand (run-led softness)."
        else:
            conclusion = "Conclusion: The build is modest; confirmation required before shifting balance view."
    elif inv_word == "draw":
        if is_offset:
            conclusion = "Conclusion: The draw occurred alongside offsetting two-way flows; avoid over-interpreting a single print."
        elif net_external < -0.50:
            conclusion = "Conclusion: The draw aligns with stronger net outflow dynamics (export pull or weaker imports)."
        elif run >= 1.0:
            conclusion = "Conclusion: The draw is consistent with stronger refinery demand (runs up)."
        else:
            conclusion = "Conclusion: The draw is modest; confirm persistence over subsequent prints."
    else:
        conclusion = "Conclusion: Flat inventories; weekly balance signal is limited."

    abnormal = _abnormality_sentence(row)

    # Next checks (clean and less confusing)
    if is_offset:
        next_checks = (
            "Next checks:\n"
            "1) Persistence: do offsetting two-way flows repeat for 2–3 weeks?\n"
            "2) Exports: check export economics (Brent–WTI, USGC margins) and destination pull.\n"
            "3) Imports: validate arrivals/timing (weather, port constraints, cargo schedules).\n"
            "4) Trend: defer to 4–8 week inventory direction for higher-conviction bias.\n"
        )
    else:
        next_checks = (
            "Next checks:\n"
            "1) Persistence: does the same driver repeat over 2–3 prints?\n"
            "2) Flow validation: reconcile imports/exports with observed differentials and logistics.\n"
            "3) Refinery context: runs vs maintenance/utilisation; confirm on product balances.\n"
            "4) Trend: confirm bias using 4–8 week inventory trajectory.\n"
        )

    return (
        f"Weekly Inventory Attribution — {d}\n\n"
        f"Headline:\n"
        f"• Inventory change: {inv_abs:.2f} mb {inv_word}\n"
        f"• Bias: {bias}\n"
        f"• Conviction: {conv} (inventory signal: {inv_signal})\n\n"
        f"What moved (WoW, mb/week):\n"
        f"• Imports: {fmt_signed(imp)}\n"
        f"• Exports: {fmt_signed(exp)}\n"
        f"• Runs:    {fmt_signed(run)}\n"
        f"• Net external (Imports−Exports): {fmt_signed(net_external)}\n\n"
        f"{driver_line}\n"
        + (f"{offset_reason}\n" if offset_reason else "")
        + "Flow configuration:\n"
        + "• " + "\n• ".join(flow_config) + "\n\n"
        + conclusion + "\n"
        + abnormal + "\n\n"
        + next_checks
    )


# =========================================================
# MAIN
# =========================================================
def main():
    ensure_dirs()
    api_key = get_api_key()

    # Fetch
    inv = fetch_eia_series(SERIES["inventory_kb"], api_key).rename(columns={"value": "inventory_kb"})
    imp = fetch_eia_series(SERIES["imports_kbd"], api_key).rename(columns={"value": "imports_kbd"})
    exp = fetch_eia_series(SERIES["exports_kbd"], api_key).rename(columns={"value": "exports_kbd"})
    run = fetch_eia_series(SERIES["runs_kbd"], api_key).rename(columns={"value": "runs_kbd"})

    # Merge weekly
    df = inv.merge(imp, on="date", how="inner") \
            .merge(exp, on="date", how="inner") \
            .merge(run, on="date", how="inner") \
            .sort_values("date")

    # Units
    df["inventory_mb"] = df["inventory_kb"] / 1000.0
    df["inventory_change_mb"] = df["inventory_mb"].diff()

    # Flows kb/d -> mb/week
    df["imports_mb_w"] = df["imports_kbd"] * 7 / 1000.0
    df["exports_mb_w"] = df["exports_kbd"] * 7 / 1000.0
    df["runs_mb_w"] = df["runs_kbd"] * 7 / 1000.0

    # WoW changes (mb/week)
    df["imports_change_mb_w"] = df["imports_mb_w"].diff()
    df["exports_change_mb_w"] = df["exports_mb_w"].diff()
    df["runs_change_mb_w"] = df["runs_mb_w"].diff()

    # z-scores (abnormality)
    df["imports_change_z"] = rolling_zscore(df["imports_change_mb_w"], ZSCORE_WINDOW)
    df["exports_change_z"] = rolling_zscore(df["exports_change_mb_w"], ZSCORE_WINDOW)
    df["runs_change_z"] = rolling_zscore(df["runs_change_mb_w"], ZSCORE_WINDOW)

    # Driver + confidence + abnormality
    drivers = df.apply(pick_driver, axis=1, result_type="expand")
    df["dominant_driver"] = drivers[0]
    df["driver_confidence"] = drivers[1]
    df["most_abnormal_flow"] = df.apply(most_abnormal_flow, axis=1)

    # Clean/trim
    df = df.dropna().reset_index(drop=True)
    if len(df) > KEEP_LAST_WEEKS:
        df = df.tail(KEEP_LAST_WEEKS).reset_index(drop=True)

    # Save CSV
    df.to_csv(OUT_CSV, index=False)

    # Prep for dashboard & note
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
    df["commentary"] = df.apply(build_refined_commentary, axis=1)
    last_dict = df.iloc[-1].to_dict()
    note = build_refined_commentary(last_dict)


    # =====================================================
    # DASHBOARD
    # =====================================================
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[
            "Weekly Inventory Change (mb) — Draw/Build",
            "Flow Attribution — Imports vs Exports (WoW change, mb/week)",
            "Refinery Runs (WoW change, mb/week)",
            "Inventory Level (mb)"
        ],
        row_heights=[0.25, 0.25, 0.20, 0.30]
    )


    fig.add_trace(
        go.Bar(
            x=df["date_str"],
            y=df["inventory_change_mb"],
            customdata=df["inventory_change_mb"].map(lambda v: f"{v:+.2f}"),
            name="Inventory change (mb)",
            hovertemplate="Date: %{x}<br>Inventory change: %{customdata} mb<extra></extra>"
        ),
        row=1, col=1
    )


    fig.add_trace(
        go.Scatter(
            x=df["date_str"],
            y=df["imports_change_mb_w"],
            customdata=df["imports_change_mb_w"].map(lambda v: f"{v:+.2f}"),
            mode="lines",
            name="Imports change (mb/week)",
            hovertemplate="Date: %{x}<br>Imports change: %{customdata} mb/week<extra></extra>"
        ),
        row=2, col=1
    )


    fig.add_trace(
        go.Scatter(
            x=df["date_str"],
            y=df["exports_change_mb_w"],
            customdata=df["exports_change_mb_w"].map(lambda v: f"{v:+.2f}"),
            mode="lines",
            name="Exports change (mb/week)",
            hovertemplate="Date: %{x}<br>Exports change: %{customdata} mb/week<extra></extra>"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["date_str"],
            y=df["runs_change_mb_w"],
            customdata=df["runs_change_mb_w"].map(lambda v: f"{v:+.2f}"),
            mode="lines",
            name="Runs change (mb/week)",
            hovertemplate="Date: %{x}<br>Runs change: %{customdata} mb/week<extra></extra>"
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["date_str"],
            y=df["inventory_mb"],
            customdata=df["inventory_mb"].map(lambda v: f"{v:.2f}"),
            mode="lines",
            name="Inventory level (mb)",
            hovertemplate="Date: %{x}<br>Inventories: %{customdata} mb<extra></extra>"
        ),
        row=4, col=1
    )



    fig.update_layout(
        title="Project 3 — Inventory Attribution Dashboard (Weekly)",
        height=1100,
        hovermode="x unified",
        hoverlabel=dict(namelength=-1)
    )



    fig.update_xaxes(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikedash="dot"
    )


    # Payload for JS commentary on hover
    payload = df[[
        "date_str",
        "inventory_mb",
        "inventory_change_mb",
        "imports_change_mb_w",
        "exports_change_mb_w",
        "runs_change_mb_w",
        "dominant_driver",
        "driver_confidence",
        "most_abnormal_flow",
        "commentary"
    ]].to_dict(orient="records")
        
        


    plot_json = fig.to_json()
    data_json = json.dumps(payload)

    # Use str.format to avoid f-string brace conflicts
    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    #chart { width: 100%; height: 1100px; }
    #commentary {
      margin-top: 16px;
      padding: 14px 16px;
      border: 1px solid #ddd;
      border-radius: 10px;
      background: #fafafa;
      white-space: pre-line;
      line-height: 1.5;
    }
    .hint { margin-top: 6px; color: #555; font-size: 13px; }
  </style>
</head>
<body>

<h2>Inventory Attribution (Weekly)</h2>
<div id="chart"></div>
<div id="commentary">Hover over the chart to view the weekly analyst note.</div>
<div class="hint">Tip: Hover anywhere — the dotted vertical line and tooltip sync across all panels.</div>

<script>
  const fig = __PLOT_JSON__;
  const rows = __DATA_JSON__;

  const byDate = {};
  rows.forEach(r => byDate[r.date_str] = r);

  Plotly.newPlot("chart", fig.data, fig.layout, {responsive: true});

  function buildNote(d) {
    const r = byDate[d];
    if (!r) return "No data for " + d;
    return r.commentary;
  }

  const lastDate = rows[rows.length - 1].date_str;
  document.getElementById("commentary").textContent = buildNote(lastDate);

  document.getElementById("chart").on("plotly_hover", e => {
    const d = e.points[0].x;
    document.getElementById("commentary").textContent = buildNote(d);
  });
</script>

</body>
</html>
"""
    html = html.replace("__PLOT_JSON__", plot_json).replace("__DATA_JSON__", data_json)


    OUT_HTML.write_text(html, encoding="utf-8")

    # Console summary
    last = df.iloc[-1]
    print(f"Saved CSV:  {OUT_CSV}")
    print(f"Saved HTML: {OUT_HTML}")
    print(
        f"Latest {last['date'].date()} | Inv chg {fmt_signed(last['inventory_change_mb'])} mb | "
        f"Driver {last['dominant_driver']} ({last['driver_confidence']})"
    )

if __name__ == "__main__":
    main()
