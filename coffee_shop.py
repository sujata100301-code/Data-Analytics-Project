"""
Boutique Coffee Shop — Operations & Waste Optimization
=======================================================
Single-file application:
  • Backend  : FastAPI (REST API + static file serving)
  • Frontend : Embedded HTML/CSS/JS (ECharts via CDN)
  • ML Model : Linear Regression demand forecaster (scikit-learn)
               predicts per-item daily demand based on day-of-week
               and hour features, enabling dynamic batch-size advice.

Run:
    pip install fastapi uvicorn scikit-learn numpy
    python coffee_shop.py

Then open: http://localhost:8000
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. ML Model — Demand Forecaster
# ---------------------------------------------------------------------------

ITEMS = [
    "Espresso", "Latte", "Cappuccino", "Cold Brew",
    "Croissant", "Muffin", "Scone", "Brownie", "Matcha",
]

# Base daily demand (units sold) per item — Mon…Sun baseline
_BASE_DEMAND: dict[str, list[float]] = {
    "Espresso":   [90, 88, 85, 92, 98, 110, 105],
    "Latte":      [130, 125, 120, 135, 145, 160, 150],
    "Cappuccino": [110, 108, 105, 112, 118, 130, 125],
    "Cold Brew":  [70,  68,  65,  72,  80,  95,  90],
    "Croissant":  [55,  52,  50,  58,  65,  75,  70],
    "Muffin":     [40,  38,  36,  42,  48,  55,  52],
    "Scone":      [25,  24,  22,  26,  30,  35,  32],
    "Brownie":    [30,  28,  27,  32,  38,  44,  40],
    "Matcha":     [60,  58,  55,  62,  70,  80,  75],
}

# Gross margin per item (%)
MARGINS: dict[str, float] = {
    "Espresso": 0.80, "Latte": 0.75, "Cappuccino": 0.75,
    "Cold Brew": 0.80, "Croissant": 0.50, "Muffin": 0.45,
    "Scone": 0.40, "Brownie": 0.45, "Matcha": 0.80,
}

# Unit price per item ($)
PRICES: dict[str, float] = {
    "Espresso": 3.50, "Latte": 5.50, "Cappuccino": 5.00,
    "Cold Brew": 5.00, "Croissant": 4.00, "Muffin": 3.50,
    "Scone": 3.00, "Brownie": 3.50, "Matcha": 5.50,
}

# Whether the item is a perishable food (not beverage)
IS_PERISHABLE: dict[str, bool] = {
    "Espresso": False, "Latte": False, "Cappuccino": False,
    "Cold Brew": False, "Croissant": True, "Muffin": True,
    "Scone": True, "Brownie": True, "Matcha": False,
}


def _build_training_data() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Construct synthetic training features and targets from baseline demand.

    Features per sample: [day_sin, day_cos, week_num_norm]
    (encode day-of-week cyclically so Mon≈Sun in the embedding space)
    Targets: one array per item.
    """
    rng = np.random.default_rng(42)
    X_rows: list[list[float]] = []
    y_rows: dict[str, list[float]] = {item: [] for item in ITEMS}

    # Simulate 52 weeks of daily data
    for week in range(52):
        for dow in range(7):  # 0=Mon … 6=Sun
            angle = 2 * math.pi * dow / 7
            day_sin = math.sin(angle)
            day_cos = math.cos(angle)
            week_norm = week / 51.0
            X_rows.append([day_sin, day_cos, week_norm])

            for item in ITEMS:
                base = _BASE_DEMAND[item][dow]
                # Add a slight upward trend over the year (+5%) and noise
                trend = base * (1 + 0.05 * week_norm)
                noise = rng.normal(0, base * 0.05)
                y_rows[item].append(max(0.0, trend + noise))

    X = np.array(X_rows, dtype=float)
    y = {item: np.array(vals, dtype=float) for item, vals in y_rows.items()}
    return X, y


class DemandForecaster:
    """Per-item linear regression models trained on synthetic historical data."""

    def __init__(self) -> None:
        X, y_dict = _build_training_data()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.models: dict[str, LinearRegression] = {}
        for item in ITEMS:
            model = LinearRegression()
            model.fit(X_scaled, y_dict[item])
            self.models[item] = model

    def predict(self, target_date: date) -> dict[str, float]:
        """Return predicted unit demand for each item on *target_date*."""
        dow = target_date.weekday()  # 0=Mon
        angle = 2 * math.pi * dow / 7
        # Use week 52 norm (latest) for future dates
        X_raw = np.array([[math.sin(angle), math.cos(angle), 1.0]])
        X_scaled = self.scaler.transform(X_raw)
        return {
            item: max(0.0, float(self.models[item].predict(X_scaled)[0]))
            for item in ITEMS
        }


# Instantiate once at startup
_forecaster = DemandForecaster()


# ---------------------------------------------------------------------------
# 2. Business Logic Helpers
# ---------------------------------------------------------------------------

def _hourly_footfall() -> list[dict[str, Any]]:
    """Representative hourly customer counts (6 AM – 7 PM)."""
    hours = [
        "6AM","7AM","8AM","9AM","10AM","11AM",
        "12PM","1PM","2PM","3PM","4PM","5PM","6PM","7PM",
    ]
    counts = [12, 38, 95, 142, 138, 74, 88, 65, 42, 24, 18, 15, 21, 14]
    return [{"hour": h, "customers": c} for h, c in zip(hours, counts)]


def _weekly_wastage(forecasted: dict[str, float]) -> list[dict[str, Any]]:
    """
    Estimate weekly wastage for perishables.
    Assume 60 % sell-through in the afternoon (post-noon) batch;
    remaining 40 % is discarded.
    """
    result = []
    for item in ITEMS:
        if not IS_PERISHABLE[item]:
            continue
        daily_demand = forecasted[item]
        # Bake daily_demand * 1.1 (10 % buffer)
        baked = daily_demand * 1.1
        sold = daily_demand * 0.60  # afternoon sell-through only (conservative)
        wasted_units = max(0.0, baked - sold) * 7  # weekly
        wasted_pct = (wasted_units / (baked * 7)) * 100 if baked > 0 else 0
        result.append({
            "item": item,
            "baked_weekly": round(baked * 7, 1),
            "sold_weekly": round(sold * 7, 1),
            "wasted_weekly": round(wasted_units, 1),
            "wastage_pct": round(wasted_pct, 1),
        })
    return result


def _recommendations(forecasted: dict[str, float]) -> list[dict[str, str]]:
    recs = []
    wastage = _weekly_wastage(forecasted)
    for w in wastage:
        if w["wastage_pct"] > 30:
            reduced = round(forecasted[w["item"]] * 0.90)
            recs.append({
                "item": w["item"],
                "action": f"Reduce morning batch by 10% → bake {reduced} units/day",
                "rationale": f"Current wastage: {w['wastage_pct']}% — above 30% threshold",
            })
    recs.append({
        "item": "Pastries (all)",
        "action": "Launch 2-for-1 Happy Hour, 3 PM – 5 PM",
        "rationale": "Afternoon footfall is <25 customers/hour; promotion can lift sell-through by ~20%",
    })
    return recs


def _productivity_index(forecasted: dict[str, float]) -> list[dict[str, Any]]:
    result = []
    for item in ITEMS:
        demand = forecasted[item]
        revenue = demand * PRICES[item]
        profit = revenue * MARGINS[item]
        # Sell-through rate: perishables suffer from afternoon drop-off
        sell_through = 0.60 if IS_PERISHABLE[item] else 0.95
        result.append({
            "item": item,
            "predicted_demand": round(demand),
            "projected_revenue": round(revenue, 2),
            "projected_profit": round(profit, 2),
            "margin_pct": round(MARGINS[item] * 100),
            "sell_through_pct": round(sell_through * 100),
        })
    result.sort(key=lambda r: r["projected_profit"], reverse=True)
    return result


# ---------------------------------------------------------------------------
# 3. FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="Coffee Shop Operations API", version="1.0.0")


@app.get("/api/footfall")
def get_footfall() -> JSONResponse:
    return JSONResponse(_hourly_footfall())


@app.get("/api/forecast")
def get_forecast(days: int = 7) -> JSONResponse:
    today = date.today()
    output = []
    for i in range(days):
        target = today + timedelta(days=i)
        pred = _forecaster.predict(target)
        output.append({
            "date": target.isoformat(),
            "day": target.strftime("%a"),
            "predictions": {k: round(v) for k, v in pred.items()},
        })
    return JSONResponse(output)


@app.get("/api/wastage")
def get_wastage() -> JSONResponse:
    forecasted = _forecaster.predict(date.today())
    return JSONResponse(_weekly_wastage(forecasted))


@app.get("/api/productivity")
def get_productivity() -> JSONResponse:
    forecasted = _forecaster.predict(date.today())
    return JSONResponse(_productivity_index(forecasted))


@app.get("/api/recommendations")
def get_recommendations() -> JSONResponse:
    forecasted = _forecaster.predict(date.today())
    return JSONResponse(_recommendations(forecasted))


@app.get("/api/kpis")
def get_kpis() -> JSONResponse:
    forecasted = _forecaster.predict(date.today())
    productivity = _productivity_index(forecasted)
    total_revenue = sum(p["projected_revenue"] for p in productivity)
    total_profit = sum(p["projected_profit"] for p in productivity)
    wastage = _weekly_wastage(forecasted)
    avg_wastage = (
        sum(w["wastage_pct"] for w in wastage) / len(wastage) if wastage else 0
    )
    footfall = _hourly_footfall()
    total_customers = sum(f["customers"] for f in footfall)
    aov = total_revenue / total_customers if total_customers else 0
    return JSONResponse({
        "total_daily_revenue": round(total_revenue, 2),
        "total_daily_profit": round(total_profit, 2),
        "average_order_value": round(aov, 2),
        "total_daily_customers": total_customers,
        "avg_wastage_rate_pct": round(avg_wastage, 1),
        "peak_hour": "9AM",
        "peak_customers": 142,
    })


# ---------------------------------------------------------------------------
# 4. Frontend — Embedded HTML/CSS/JS
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Coffee Shop Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,"Segoe UI",system-ui,sans-serif;background:#f7f8fa;color:#1f2328;font-size:14px;line-height:1.6}
  header{background:#1f2328;color:#fff;padding:16px 32px;display:flex;align-items:center;gap:12px}
  header h1{font-size:18px;font-weight:600}
  header span{font-size:13px;color:#8b949e}
  .kpi-bar{display:flex;gap:16px;padding:20px 32px;flex-wrap:wrap}
  .kpi{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px 20px;flex:1;min-width:140px}
  .kpi .label{font-size:12px;color:#57606a;text-transform:uppercase;letter-spacing:.5px}
  .kpi .value{font-size:26px;font-weight:700;color:#1f2328;margin-top:2px}
  .kpi .sub{font-size:11px;color:#57606a;margin-top:2px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:0 32px 16px}
  .card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:16px}
  .card h2{font-size:14px;font-weight:600;margin-bottom:12px;color:#1f2328}
  .chart{width:100%;height:280px}
  .full{grid-column:1/-1}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th{text-align:left;padding:6px 10px;border-bottom:2px solid #e5e7eb;color:#57606a;font-weight:600;font-size:12px;text-transform:uppercase}
  td{padding:6px 10px;border-bottom:1px solid #f3f4f6}
  tr:last-child td{border-bottom:none}
  .badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}
  .badge.high{background:#dbeafe;color:#1d4ed8}
  .badge.med{background:#fef3c7;color:#92400e}
  .badge.low{background:#fee2e2;color:#991b1b}
  .rec-list{list-style:none;padding:0}
  .rec-list li{padding:10px 12px;border-left:3px solid #3b82d4;background:#f0f6ff;border-radius:0 6px 6px 0;margin-bottom:8px}
  .rec-list li .item{font-weight:600;font-size:13px}
  .rec-list li .action{color:#1d4ed8;margin:2px 0;font-size:13px}
  .rec-list li .rationale{color:#57606a;font-size:12px}
  @media(max-width:700px){.grid{grid-template-columns:1fr}.kpi-bar{flex-direction:column}}
</style>
</head>
<body>

<header>
  <div>
    <h1>☕ Coffee Shop Operations Dashboard</h1>
    <span>Waste Optimization &amp; Demand Forecasting</span>
  </div>
</header>

<div class="kpi-bar" id="kpi-bar">
  <div class="kpi"><div class="label">Daily Revenue</div><div class="value" id="kpi-rev">—</div><div class="sub">Projected today</div></div>
  <div class="kpi"><div class="label">Daily Profit</div><div class="value" id="kpi-profit">—</div><div class="sub">After COGS</div></div>
  <div class="kpi"><div class="label">Avg Order Value</div><div class="value" id="kpi-aov">—</div><div class="sub">Per customer</div></div>
  <div class="kpi"><div class="label">Customers Today</div><div class="value" id="kpi-cust">—</div><div class="sub">Footfall estimate</div></div>
  <div class="kpi"><div class="label">Wastage Rate</div><div class="value" id="kpi-waste" style="color:#d94f4f">—</div><div class="sub">Perishables discarded</div></div>
  <div class="kpi"><div class="label">Peak Hour</div><div class="value" id="kpi-peak">—</div><div class="sub" id="kpi-peak-sub">—</div></div>
</div>

<div class="grid">
  <div class="card">
    <h2>Hourly Customer Footfall</h2>
    <div class="chart" id="chart-footfall"></div>
  </div>
  <div class="card">
    <h2>Perishable Wastage (weekly)</h2>
    <div class="chart" id="chart-wastage"></div>
  </div>
  <div class="card">
    <h2>Item Revenue vs. Profit (today)</h2>
    <div class="chart" id="chart-profit"></div>
  </div>
  <div class="card">
    <h2>7-Day Demand Forecast — Lattes &amp; Croissants</h2>
    <div class="chart" id="chart-forecast"></div>
  </div>
  <div class="card full">
    <h2>Productivity &amp; Margin Index</h2>
    <table id="prod-table">
      <thead><tr><th>Item</th><th>Forecast Units</th><th>Revenue ($)</th><th>Profit ($)</th><th>Margin</th><th>Sell-Through</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>
  <div class="card full">
    <h2>Recommended Actions</h2>
    <ul class="rec-list" id="rec-list"></ul>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const fmt = (n, prefix='$') => prefix + n.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
const fmtPct = n => n.toFixed(1) + '%';

function badgeClass(pct) {
  if (pct >= 70) return 'high';
  if (pct >= 50) return 'med';
  return 'low';
}

async function fetchJSON(url) {
  const r = await fetch(url);
  return r.json();
}

function initChart(id, option) {
  const chart = echarts.init(document.getElementById(id));
  chart.setOption(option);
  window.addEventListener('resize', () => chart.resize());
  return chart;
}

(async () => {
  const [kpis, footfall, wastage, productivity, forecast, recs] = await Promise.all([
    fetchJSON('/api/kpis'),
    fetchJSON('/api/footfall'),
    fetchJSON('/api/wastage'),
    fetchJSON('/api/productivity'),
    fetchJSON('/api/forecast?days=7'),
    fetchJSON('/api/recommendations'),
  ]);

  // --- KPIs ---
  $('kpi-rev').textContent    = fmt(kpis.total_daily_revenue);
  $('kpi-profit').textContent = fmt(kpis.total_daily_profit);
  $('kpi-aov').textContent    = fmt(kpis.average_order_value);
  $('kpi-cust').textContent   = kpis.total_daily_customers;
  $('kpi-waste').textContent  = fmtPct(kpis.avg_wastage_rate_pct);
  $('kpi-peak').textContent   = kpis.peak_hour;
  $('kpi-peak-sub').textContent = kpis.peak_customers + ' customers/hr';

  // --- Footfall bar chart ---
  initChart('chart-footfall', {
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: footfall.map(f => f.hour) },
    yAxis: { type: 'value', name: 'Customers' },
    series: [{ type: 'bar', data: footfall.map(f => f.customers),
      itemStyle: { color: p => p.dataIndex >= 2 && p.dataIndex <= 4 ? '#3b82d4' : '#93c5fd' } }],
    grid: { left: 40, right: 10, top: 20, bottom: 30 },
  });

  // --- Wastage pie chart ---
  initChart('chart-wastage', {
    tooltip: { trigger: 'item', formatter: '{b}: {c} units ({d}%)' },
    legend: { bottom: 0, textStyle: { fontSize: 11 } },
    series: [{
      type: 'pie', radius: ['38%', '65%'], center: ['50%', '45%'],
      label: { formatter: '{b}\n{d}%', fontSize: 11 },
      data: wastage.map(w => ({ name: w.item, value: w.wasted_weekly })),
    }],
  });

  // --- Revenue vs Profit bar chart ---
  const items = productivity.map(p => p.item);
  initChart('chart-profit', {
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { fontSize: 11 } },
    xAxis: { type: 'category', data: items, axisLabel: { fontSize: 10, rotate: 30 } },
    yAxis: { type: 'value', name: '$' },
    series: [
      { name: 'Revenue', type: 'bar', data: productivity.map(p => p.projected_revenue), color: '#3b82d4' },
      { name: 'Profit',  type: 'bar', data: productivity.map(p => p.projected_profit),  color: '#7c5cd8' },
    ],
    grid: { left: 40, right: 10, top: 40, bottom: 60 },
  });

  // --- 7-day demand forecast line chart ---
  const days = forecast.map(f => f.day + ' ' + f.date.slice(5));
  initChart('chart-forecast', {
    tooltip: { trigger: 'axis' },
    legend: { top: 0, textStyle: { fontSize: 11 } },
    xAxis: { type: 'category', data: days, axisLabel: { fontSize: 10 } },
    yAxis: { type: 'value', name: 'Units' },
    series: [
      { name: 'Latte',     type: 'line', smooth: true, data: forecast.map(f => f.predictions['Latte']),     color: '#3b82d4', symbolSize: 6 },
      { name: 'Croissant', type: 'line', smooth: true, data: forecast.map(f => f.predictions['Croissant']), color: '#e07b39', symbolSize: 6, lineStyle: { type: 'dashed' } },
    ],
    grid: { left: 40, right: 10, top: 40, bottom: 30 },
  });

  // --- Productivity table ---
  const tbody = document.querySelector('#prod-table tbody');
  productivity.forEach(p => {
    const tr = document.createElement('tr');
    const stClass = badgeClass(p.sell_through_pct);
    const mgClass = badgeClass(p.margin_pct);
    tr.innerHTML = `
      <td><strong>${p.item}</strong></td>
      <td>${p.predicted_demand}</td>
      <td>${fmt(p.projected_revenue)}</td>
      <td>${fmt(p.projected_profit)}</td>
      <td><span class="badge ${mgClass}">${p.margin_pct}%</span></td>
      <td><span class="badge ${stClass}">${p.sell_through_pct}%</span></td>
    `;
    tbody.appendChild(tr);
  });

  // --- Recommendations ---
  const recList = $('rec-list');
  recs.forEach(r => {
    const li = document.createElement('li');
    li.innerHTML = `
      <div class="item">${r.item}</div>
      <div class="action">→ ${r.action}</div>
      <div class="rationale">${r.rationale}</div>
    `;
    recList.appendChild(li);
  });
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def get_dashboard() -> HTMLResponse:
    return HTMLResponse(_HTML)


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
