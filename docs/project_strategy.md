# M5 Forecasting — Learning Strategy & Project Roadmap

**Author:** Gurjinder Kaur — Senior Data Scientist, CVS Health (Front Store Demand Forecasting)
**Purpose:** Use the M5 competition dataset as a laboratory to build deep forecasting expertise — both the science and the engineering — applicable to millions-of-SKUs retail demand planning.
**Started:** April 2026

---

## Why M5

The M5 dataset (Walmart, ~30,490 item-store combinations across 10 stores in 3 US states, 1,941 days of daily sales) is a scaled-down version of the exact problem you solve at CVS. It has the same characteristics: extreme sparsity at the item-store level, hierarchical product/geography structure, promotional and calendar effects (SNAP, holidays), and price variation. The difference is scale — M5 is small enough that you can validate ideas exhaustively before applying them to millions of SKUs.

You are not trying to win this competition. You are building a decision-making framework for production forecasting systems.

---

## Data Inventory

| File | Description | Shape (approx) |
|------|-------------|-----------------|
| `sales_train_evaluation.csv` | Daily unit sales, item-store level, d_1 through d_1941 | 30,490 rows x 1,947 cols |
| `calendar.csv` | Date mapping, events, SNAP flags | 1,969 rows x 14 cols |
| `sell_prices.csv` | Weekly sell prices by store-item | ~6.8M rows x 4 cols |
| `sales_train_validation.csv` | Same as evaluation but ends 28 days earlier (d_1 to d_1913) | 30,490 rows x 1,919 cols |
| `sample_submission.csv` | Submission format for the competition | 60,980 rows |

---

## Phase Overview

### Phase 0: Messy Exploration (Week 1–2)
No clean code. No repo structure. Just intuition-building through aggressive EDA. Scratch notebooks, a dated exploration log, and lots of questions. The goal is to understand the data well enough to make informed design decisions later.

**Deliverable:** `docs/exploration_log.md`, initial hypotheses about demand patterns.

### Phase 1: Data Understanding & Decomposition (Week 2–4)
Systematic analysis of signal-to-noise at each hierarchy level. Decomposition of demand into trend, seasonality, calendar effects, and noise. Sparsity and intermittency profiling. This phase directly informs which models are even appropriate.

**Deliverable:** `docs/learnings/data_decomposition.md`, diagnostic visualizations, demand taxonomy.

### Phase 2: Repo Structure & Engineering Setup (Week 4–5)
Migrate reusable code from notebooks into `src/`. Set up config-driven experiment management, reproducible environments, and testing. Establish the ADR (Architecture Decision Record) practice.

**Deliverable:** Clean repo structure, `pyproject.toml`, first ADRs in `docs/decisions/`.

### Phase 3: Statistical Baselines (Week 5–7)
Implement classical forecasting methods — naive, ETS, Theta, AutoARIMA, Croston/TSB — and understand which demand patterns each handles well. Build the cross-validation harness.

**Deliverable:** `docs/learnings/statistical_baselines.md`, baseline accuracy table, CV framework in `src/evaluation/`.

### Phase 4: ML Models & Feature Engineering (Week 7–10)
LightGBM with direct multi-step forecasting. Deep feature engineering: lags, rolling statistics, calendar encodings, price features, hierarchical aggregates. Rigorous time series cross-validation.

**Deliverable:** `docs/learnings/ml_models.md`, `src/features/`, `src/models/`, feature importance analysis.

### Phase 5: Evaluation & Metrics Deep Dive (Week 10–11)
Implement WRMSSE, MASE, sMAPE, scaled pinball loss, CRPS. Understand how metric choice changes model selection and business outcomes. Connect metrics to inventory planning decisions.

**Deliverable:** `docs/learnings/evaluation_metrics.md`, `src/evaluation/metrics.py`, metric comparison analysis.

### Phase 6: Probabilistic & Hierarchical Forecasting (Week 11–14)
Quantile forecasts via conformal prediction and quantile regression. Hierarchical reconciliation (bottom-up, top-down, MinT). Understand when reconciliation improves accuracy vs. just ensures coherence.

**Deliverable:** `docs/learnings/probabilistic_forecasting.md`, `docs/learnings/hierarchical_reconciliation.md`, `src/reconciliation/`.

### Phase 7: Ensembling & Production Patterns (Week 14–16)
Model combination strategies. Config-driven experiment tracking. Logging with MLflow. Testing pipeline code. Reproducibility practices.

**Deliverable:** `docs/learnings/ensembling.md`, production-ready pipeline, final project writeup.

---

## Repo Structure (Target — Phase 2+)

```
m5-forecasting/
├── README.md
├── pyproject.toml
├── docs/
│   ├── project_strategy.md          # This file
│   ├── exploration_log.md           # Dated entries from Phase 0
│   ├── decisions/                   # Architecture Decision Records
│   │   ├── 001-granularity-choice.md
│   │   ├── 002-cv-strategy.md
│   │   └── ...
│   └── learnings/                   # Structured writeups per topic
│       ├── data_decomposition.md
│       ├── statistical_baselines.md
│       └── ...
├── notebooks/                       # Exploration only — never called by pipeline
├── data/                            # Raw M5 data (gitignored)
├── src/
│   ├── data/                        # Loading, cleaning, reshaping
│   ├── features/                    # Feature engineering modules
│   ├── models/                      # Model wrappers and training logic
│   ├── evaluation/                  # Metrics, CV harness, comparison tools
│   ├── reconciliation/              # Hierarchical reconciliation
│   └── pipeline/                    # Orchestration scripts
├── configs/                         # YAML experiment configs
├── tests/                           # Unit and integration tests
└── experiments/                     # Logged results (gitignored or LFS)
```

---

## Documentation Practice

Every significant learning gets a structured writeup in `docs/learnings/` with four sections:

1. **Context** — What problem were you solving? What question were you trying to answer?
2. **What I tried** — Methods, configurations, code references.
3. **What I found** — Results, surprises, failures. Include numbers and plots.
4. **Implications for production** — How does this change how you would design a system at CVS scale? This is the most important section.

Every significant design choice gets an ADR in `docs/decisions/` using this template:

```
# ADR-NNN: [Title]

## Status: [Proposed | Accepted | Deprecated | Superseded]

## Context
What is the issue that we're seeing that motivates this decision?

## Decision
What is the change we're proposing?

## Consequences
What becomes easier or harder because of this change?
```

---

## Primary Reference Book

**"Mastering Modern Time Series Forecasting" — Valery Manokhin, PhD (2025/2026)**
North Star Academic Press. 762 pages, Python throughout. You have this book already.

This is the primary technical reference for the project. The chapter-to-phase mapping below tells you when to read each chapter — don't try to read it front to back. Use it as a companion that you consult when you are about to start each phase.

| Chapter | Topic | Read For Phase |
|---------|-------|---------------|
| 1 | Evolution and principles of forecasting | Background context — skim anytime |
| 2 | Forecastability — ADI/CV2, entropy, predictability limits | Phase 0 |
| 3 | Metrics — point, probabilistic, hierarchical, FVA | Phase 5 |
| 4 | ARIMA, SARIMA, AutoARIMA — theory and Python | Phase 3 |
| 5 | ETS, exponential smoothing — theory and Python | Phase 3 |
| 6 | CV strategies, forecast stability, drift detection | Phases 3–4 |
| 7 | Feature engineering — statistical, decomposition, spectral | Phase 4 |
| 8 | Advanced ML — GBDT, global models, lag/window features | Phase 4 |
| 9 | Deep learning — RNN, LSTM, N-BEATS, DLinear | Phase 7+ |
| 10 | Transformers — PatchTST, TFT, critical analysis | Phase 7+ |
| 11 | Foundation models — Chronos, TimeGPT, Moirai | Phase 7+ |

---

## Guiding Principles

- **Notebooks are disposable. Modules are permanent.** Anything you use twice goes into `src/`.
- **Document the why, not the what.** Code shows what you did. ADRs and learnings show why.
- **Simple models first.** You cannot evaluate a complex model without a solid baseline.
- **Metric choice is a business decision.** There is no universally correct accuracy metric.
- **At scale, you forecast populations of series, not individual series.** Build taxonomies, not per-SKU models.
- **Reproducibility is non-negotiable.** Every experiment should be re-runnable from a config file.
