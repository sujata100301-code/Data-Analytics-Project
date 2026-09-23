# ☕ Boutique Coffee Shop — Operations & Waste Optimization

A self-contained, single-file Python web application that combines a **FastAPI backend**, an **embedded HTML/JS frontend**, and a **scikit-learn ML demand forecasting model** to help a specialty coffee shop reduce perishable waste and maximize revenue.

---

## Features

| Layer | Technology | Purpose |
|---|---|---|
| Backend | FastAPI + Uvicorn | REST API serving all operational data |
| Frontend | Embedded HTML / ECharts (CDN) | Live dashboard with 4 charts + KPI bar |
| ML Model | scikit-learn `LinearRegression` | Per-item daily demand forecasting |

### Dashboard panels
- **KPI Bar** — Daily revenue, profit, AOV, customer count, wastage rate, peak hour
- **Hourly Footfall Chart** — Bar chart of customer distribution across 14 hours
- **Perishable Wastage Donut** — Weekly waste breakdown by food item
- **Revenue vs. Profit Bar Chart** — Side-by-side comparison across all 9 menu items
- **7-Day Demand Forecast** — Line chart for Latte & Croissant projections
- **Productivity & Margin Table** — Full item table with colour-coded margin/sell-through badges
- **Recommended Actions** — ML-driven operational suggestions (batch reduction, happy hour)

---

## Project Structure

```
coffee_shop.py      # Entire application — backend + frontend + ML
requirements.txt    # Python dependencies
README.md           # This file
PROJECT_REPORT.md   # Full technical & business report
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- pip

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install fastapi==0.111.0 uvicorn[standard]==0.29.0 scikit-learn==1.4.2 numpy==1.26.4
```

### 3. Run the application

```bash
python coffee_shop.py
```

The server starts on `http://localhost:8000`.

### 4. Open the dashboard

Navigate to **[http://localhost:8000](http://localhost:8000)** in any modern browser.

---

## API Endpoints

All endpoints return JSON. No authentication required.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the HTML dashboard |
| `GET` | `/api/kpis` | 6 headline KPIs for today |
| `GET` | `/api/footfall` | Hourly customer counts (6 AM – 7 PM) |
| `GET` | `/api/forecast?days=N` | N-day demand forecast per item (default: 7) |
| `GET` | `/api/wastage` | Weekly wastage stats for perishable items |
| `GET` | `/api/productivity` | Revenue, profit, margin & sell-through per item |
| `GET` | `/api/recommendations` | Prioritised operational action list |

### Example — forecast for next 3 days
```bash
curl http://localhost:8000/api/forecast?days=3
```

```json
[
  {
    "date": "2024-07-15",
    "day": "Mon",
    "predictions": { "Latte": 133, "Croissant": 56, ... }
  },
  ...
]
```

---

## ML Model Details

The `DemandForecaster` class trains **one `LinearRegression` model per menu item** (9 models total) at application startup.

**Features (3 per sample):**

| Feature | Description |
|---|---|
| `day_sin` | `sin(2π × day_of_week / 7)` — cyclic day encoding |
| `day_cos` | `cos(2π × day_of_week / 7)` — cyclic day encoding |
| `week_norm` | Normalised week number (0–1) capturing annual trend |

**Training data:** 52 weeks × 7 days = 364 samples of synthetic demand per item, incorporating ±5% Gaussian noise and a +5% annual growth trend.

All features are standardised with `StandardScaler` before fitting.

---

## Menu Items & Configuration

| Item | Price | Margin | Perishable |
|---|---|---|---|
| Espresso | $3.50 | 80% | No |
| Latte | $5.50 | 75% | No |
| Cappuccino | $5.00 | 75% | No |
| Cold Brew | $5.00 | 80% | No |
| Matcha | $5.50 | 80% | No |
| Croissant | $4.00 | 50% | **Yes** |
| Muffin | $3.50 | 45% | **Yes** |
| Scone | $3.00 | 40% | **Yes** |
| Brownie | $3.50 | 45% | **Yes** |

To adjust prices, margins, or demand baselines, edit the `PRICES`, `MARGINS`, and `_BASE_DEMAND` dictionaries at the top of [`coffee_shop.py`](coffee_shop.py).

---

## Key Business Rules

- **Bake buffer:** Each perishable item is baked at `forecast × 1.10` (10% safety buffer).
- **Afternoon sell-through assumption:** 60% of perishable demand is sold post-noon (conservative).
- **Wastage threshold:** Items with >30% weekly wastage trigger a batch-reduction recommendation.
- **Happy Hour trigger:** Always recommended when afternoon footfall is below 25 customers/hour.

---

## License

MIT — free to use and adapt for personal or commercial purposes.
