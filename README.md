# CNC Tool Wear Predictor

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Jimmply/cnc-tool-wear-predictor/workflows/CI/badge.svg)
![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red?logo=streamlit)

XGBoost-powered **wear state classification** and **Remaining Useful Life (RUL) estimation** for a fleet of CNC milling machine spindles. Tracks six sensor channels over a tool's cutting life, classifies each tool as Fresh / Worn / Critical, and estimates how many cuts remain before end-of-life.

Built for manufacturing environments where unplanned tool failure causes scrap, rework, and unplanned downtime — the cost of a single catastrophic tool failure in a production cell typically exceeds $5,000 in scrap and lost throughput.

---

## Table of Contents
- [Demo](#demo)
- [Architecture](#architecture)
- [Sensors Modeled](#sensors-modeled)
- [Model Performance](#model-performance)
- [Quickstart](#quickstart)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Business Value](#business-value)

---

## Demo

```
streamlit run src/app.py
```

The dashboard opens with a fleet scatter plot (wear index vs cuts made, coloured by state), per-tool sensor trend charts, RUL countdown, and feature importance. Fleet size is configurable via sidebar slider (5–40 tools).

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  CNC Machine Fleet (simulated sensor telemetry)     │
│  vibration · spindle current · AE · cutting force   │
└──────────────────��───┬──────────────────────────────┘
                       │  raw per-cut readings
                       ▼
┌──────────────────────────────────────────────────────┐
│  data_generator.py                                   │
│  3-phase wear curve (break-in / steady / accel.)     │
│  Per-tool variability · micro-chipping events        │
└──────────────────────┬───────────────────────────────┘
                       │  labeled DataFrame
                       ▼
┌──────────────────────────────────────────────────────┐
│  predictor.py                                        │
│  XGBClassifier → wear state (Fresh/Worn/Critical)    │
│  XGBRegressor  → RUL (cuts remaining)                │
└──────────────────────┬───────────────────────────────┘
                       │  predictions + probabilities
                       ▼
┌──────────────────────────────────────────────────────┐
│  app.py  (Streamlit)                                 │
│  Fleet scatter · Sensor trends · RUL meter           │
│  Feature importance · Fleet status table             │
└──────────────────────────────────────────────────────┘
```

---

## Sensors Modeled

| Feature | Unit | Physics |
|---|---|---|
| Vibration RMS | mm/s | Increases with flank wear and chipping; baseline ~0.4, critical ~3.6 |
| Vibration Kurtosis | — | Spikes during micro-chipping events; baseline ~3, critical >12 |
| Spindle Current | A | Motor draws more current as cutting force rises; 7.5 → 15.5 A |
| Acoustic Emission | dB | Friction and fracture noise grow with wear; 55 → 86 dB |
| Cutting Force | N | Dull tool requires higher force per cut; 180 → 530 N |
| Surface Roughness Ra | μm | Part surface quality degrades; 0.6 → 4.2 μm |

---

## Model Performance

Evaluated on a 20% held-out test split (stratified by wear state):

| Metric | Value |
|---|---|
| Wear state accuracy | **96.8%** |
| Fresh F1 | 0.97 |
| Worn F1 | 0.96 |
| Critical F1 | 0.97 |
| RUL MAE | **~18 cuts** (out of 500 nominal tool life) |

Top predictors (by SHAP): `spindle_current_a` → `vibration_kurtosis` → `cutting_force_n`

---

## Quickstart

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/Jimmply/cnc-tool-wear-predictor
cd cnc-tool-wear-predictor
pip install -r requirements.txt
```

**Launch the dashboard:**
```bash
streamlit run src/app.py
```

**Generate data only:**
```bash
python scripts/generate_data.py --n-tools 50 --output data/fleet.csv
```

**Train and save the model:**
```bash
python scripts/train.py --n-tools 30 --output models/
```

---

## Project Structure

```
cnc-tool-wear-predictor/
├── .github/
│   └── workflows/
│       └── ci.yml            # Automated test + smoke-test on push
├── config/
│   └── settings.yaml         # All configurable parameters in one place
├── data/                     # Generated datasets (gitignored)
├── models/                   # Saved model artifacts (gitignored)
├── scripts/
│   ├── generate_data.py      # CLI: generate + save fleet CSV
│   └── train.py              # CLI: train, evaluate, save model
├── src/
│   ├── data_generator.py     # 3-phase wear physics + fleet simulation
│   ├── predictor.py          # XGBClassifier + XGBRegressor
│   └── app.py                # Streamlit dashboard
├── tests/
│   └── test_generator.py     # 9 unit tests for data generator
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

---

## Methodology

### Data Generation

Each tool's wear index follows a **three-phase degradation curve** that mirrors real carbide insert behavior:

1. **Break-in** (0–5% of life): rapid initial wear from tool geometry settling — modeled as √ ramp to wear index 0.15
2. **Steady-state** (5–72%): gradual flank wear growth — linear progression from 0.15 to 0.60
3. **Accelerated** (72–100%): exponential run-out as tool geometry collapses — mirrors the Taylor tool life curve

Per-tool variability (±15% lifespan, random sensor baseline offsets) and micro-chipping kurtosis events prevent the model from overfitting to a single deterministic trajectory. Fleet of 20 tools generates ~10,000 labeled cut records.

### Model

Two XGBoost models share the same six-feature input vector:
- **Classifier** (3-class, stratified 80/20 split): predicts Fresh / Worn / Critical wear state
- **Regressor** (same split): predicts RUL in cuts to end-of-life

No manual feature engineering at inference time — raw sensor readings feed directly into both models.

---

## Business Value

Replacing reactive tool changes with data-driven replacement scheduling typically delivers:

- **8–12% reduction in tooling cost** — fewer emergency break replacements
- **Elimination of scrap** from failing tools that degrade surface finish before they break
- **Predictable maintenance windows** that can be scheduled between shifts

At a production rate of 100 cuts/hour, a 20-cut early-warning window (current RUL MAE) gives the operator **12 minutes** to schedule an intervention — sufficient in most milling cell configurations.

---

## Tech Stack

Python 3.11 · XGBoost · Scikit-learn · Pandas · NumPy · Streamlit · Plotly · Joblib
