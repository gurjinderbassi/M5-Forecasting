# Phase 0: Messy Exploration — Study Guide & Implementation Plan

**Duration:** Week 1–2
**Goal:** Build raw intuition about the M5 data before making any modeling or engineering decisions. Understand the shape, scale, sparsity, and structure of the data deeply enough that you can make informed choices in every subsequent phase.

---

## Part 1: Study Material — Read Before You Code

The references below are selected for practical relevance to retail demand forecasting at scale. They are ordered by priority — start at the top. You do not need to finish everything before touching data, but read at least the first three before opening a notebook.

### 1.1 Foundational Understanding of Retail Demand Data

**"Forecasting: Principles and Practice — the Pythonic Way" — Hyndman, Athanasopoulos, Garza, Challu, Mergenthaler, Olivares (2025)**
Free online: https://otexts.com/fpppy/

This is the Python edition of the standard Hyndman & Athanasopoulos textbook, co-produced with the Nixtla team. Same chapter structure and conceptual depth as the R edition (FPP3), but all code is in Python using the nixtlaverse libraries (statsforecast, mlforecast, hierarchicalforecast) — the same tools you will use throughout this project. It also adds two chapters on neural networks and foundation models not present in the R edition. Use this over FPP3 since the code is directly applicable.

Read chapters 1–3 (time series patterns, decomposition) and chapter 5 (benchmark methods, residual diagnostics) before starting your exploration. The R edition (https://otexts.com/fpp3/) remains useful as a reference if you want to cross-check an explanation, since it is more mature and has more community annotations.

Why it matters for Phase 0: Chapter 2 covers exactly how to look at time series data — what patterns to look for, how to think about seasonality, trend, and cycles, and how the choice of time granularity changes what you see.

**"Supply Chain Forecasting" — Nicolas Vandeput**
Vandeput is a supply chain data scientist who writes practically about demand forecasting for retail and CPG. Search his name on Medium and LinkedIn — he has published extensively on demand classification, intermittent demand, and forecasting at scale, closer in tone to a practitioner blog than a textbook. His writing is worth reading for the CVS-scale mindset: he thinks in terms of thousands of SKUs and operational constraints, not single-series academic examples. Be aware that his specific publication titles and formats have changed over time — verify what you find before citing it.

Why it matters for Phase 0: His writing is useful for the practitioner mindset — how to think about forecasting as an operational problem at scale, not as a modeling exercise. The demand classification framework itself (smooth, erratic, intermittent, lumpy) comes from Boylan & Syntetos, covered in section 1.3 below.

**"Mastering Modern Time Series Forecasting" — Valery Manokhin, PhD (2025/2026)**
North Star Academic Press. Python throughout. You have this book already.

For Phase 0, read **Chapter 2: The Forecastability of Time Series**. This chapter directly operationalizes the question you are trying to answer in Phase 0: how much signal is actually in a series before you try to model it? It covers the ADI/CV2 framework (section 2.4.1), variance ratio as a predictability bound, ACF/PACF as forecastability signals, and entropy-based complexity measures. The forecastability framing — asking "can this series be forecast at all?" before asking "how should I forecast it?" — is exactly the right mental model for Phase 0 at scale.

Chapters for later phases: Chapter 3 (metrics) → Phase 5. Chapters 4–5 (ARIMA, ETS) → Phase 3. Chapter 6 (cross-validation, drift detection) → Phases 3–4. Chapters 7–8 (feature engineering, gradient boosting, global ML models) → Phase 4. Chapters 9–11 (deep learning, transformers, foundation models) → Phase 7+. Treat this as the primary technical reference throughout the project.

### 1.2 The M5 Competition — What the Winners Learned

**"The M5 accuracy competition: Results, findings, and conclusions" — Makridakis, Spiliotis, Assimakopoulos (2022)**
Published in International Journal of Forecasting, Vol 38, Issue 4.

This is the official post-competition paper. Read it for context on how different method families performed, but hold off on internalizing the conclusions until after you have formed your own hypotheses from the data. Come back to this paper after completing your exploration and compare your intuitions to what the competition revealed.

**"M5 Competition Uncertainty — 1st Place Solution" — Monsauce (Kaggle)**
**"M5 Accuracy — Top Solutions Discussion Thread" (Kaggle)**

Read the top 5 solution summaries on Kaggle (not the code, the writeups). Focus on what features they engineered, what granularity they chose, and how they handled sparse series. Don't try to replicate their pipelines — extract their thinking.

### 1.3 Understanding Intermittent Demand

**"Intermittent Demand Forecasting: Context, Methods and Applications" — John Boylan & Aris Syntetos (2021)**
Wiley publication. This is the authoritative practical reference on forecasting sparse/intermittent demand. Chapters 1–4 cover the ADI (Average Demand Interval) and CV-squared framework for classifying demand patterns, which is the single most useful diagnostic you can run in Phase 0.

**"Forecast value added analysis" — concept from Boylan & Syntetos**
The idea that every step in your forecasting process should demonstrably improve accuracy compared to a naive approach. Apply this thinking even during exploration: if you can't visually see a pattern, a model probably can't learn it either.

### 1.4 Practical Scale Thinking

**"Principles of Forecasting: A Handbook for Researchers and Practitioners" — J. Scott Armstrong (2001)**
An older handbook, but Armstrong's principles (keep it simple, be conservative, combine forecasts) have survived decades of validation. Read the executive summary and the "Golden Rule of Forecasting" paper (Armstrong, Green, Graefe — 2015). The core message: complexity does not automatically improve forecasts, and at large scale, simple methods often win in aggregate.

**Nixtla Blog & Documentation — https://nixtlaverse.nixtla.io/**
Nixtla (makers of statsforecast, mlforecast, hierarchicalforecast) publishes practical tutorials on scaling forecasting to millions of series. Their blog posts on "Why you should use global models" and "Cross-learning for time series" are directly relevant to both M5 and CVS-scale work.

### 1.5 Papers Worth Skimming (Not Deep-Reading)

- **"Why do we need another time series benchmarking dataset?"** — the original M5 proposal paper by Makridakis et al. Gives context on why M5 was designed the way it was.
- **"Global models for time series forecasting: A simulation study"** — Montero-Manso & Hyndman (2021). Explains when and why global (cross-learning) models beat local (per-series) models. Critical background for Phase 4 but worth skimming now to set expectations.
- **"Do we really need deep learning models for time series forecasting?"** — Zeng et al. (2023). The controversial "linear models beat transformers" paper. Useful reality check on hype.

---

## Part 2: The M5 Data — What You Are Working With

Before you code, understand the structure on paper. Open each CSV file and examine its columns, then answer the questions below yourself.

### 2.1 Hierarchy

The M5 data has a natural hierarchical structure. Your first task is to map it out:

- Open `sales_train_evaluation.csv` and look at the ID columns. What are the grouping dimensions?
- How many levels of aggregation can you construct from these dimensions? List them from most granular (bottom) to most aggregate (top).
- How many unique series exist at each level?
- How does this compare to the hierarchy you work with at CVS (SKU / store / category / region / banner / national)?

### 2.2 Time Span

- What date does the data start and end? (Use the `calendar.csv` to map day columns to actual dates.)
- How many total days of data are there?
- What is the forecast horizon used in the competition?
- What is the difference between `sales_train_validation.csv` and `sales_train_evaluation.csv`?

### 2.3 External Variables

Examine `calendar.csv` and `sell_prices.csv`:

- What external information is available in the calendar? List the columns and understand what each one represents.
- How is the price data structured? At what granularity (daily? weekly? monthly?) and what does that imply about how you join it to sales?
- How large is the price dataset and what does that tell you about data density?

---

## Part 3: Exploration Agenda — What to Actually Do

Work in scratch notebooks. Don't optimize for clean code. Optimize for speed of learning. Every finding goes in `docs/exploration_log.md` with a date stamp.

### 3.1 Data Shape and Completeness (~2 hours)

Questions to answer:

- How many unique items, stores, departments, categories, states?
- What is the date range? Are there any gaps in the calendar?
- What fraction of item-store series have non-zero sales from d_1 onward? How many items appear to have been introduced later? How would you detect this?
- What is the overall ratio of zero-sales observations to non-zero at the item-store-day level?
- What does the sell_prices data look like? How many unique prices exist per item? What is the distribution of price change frequency?

What you are building: a mental model of data completeness. At CVS, this translates to knowing which SKUs have enough history to model and which are new/discontinued.

### 3.1b Data Continuity and the Zero vs. Missing Problem (~3 hours)

This section deserves its own block because it is one of the most consequential data quality issues in production demand forecasting — and one of the most commonly ignored in Kaggle-style work. At CVS with millions of SKUs, getting this wrong contaminates every downstream feature and model.

**The core distinction:** A zero in the sales column can mean two fundamentally different things: (a) the item was on the shelf and available, but no customer bought it that day (true zero demand), or (b) the item was not available — out of stock, not yet introduced, discontinued, or the store was closed. These require completely different treatment. True zeros are valid training data. Availability-driven zeros are missing data that should be excluded or imputed, not trained on.

**Calendar continuity investigation:**

- Does the calendar file contain an unbroken sequence of dates from start to end? Check for gaps — are there any missing dates?
- Are there days when stores were likely closed (e.g., Christmas Day, Thanksgiving)? How can you detect this from the data itself without external knowledge?
- For a given store, are there any periods where ALL items show zero sales simultaneously? What would that indicate?
- Plot total daily sales per store over time. Look for days that drop to zero or near-zero. Are these isolated or do they form patterns?

**Item availability investigation:**

- For each item-store series, identify the "active window" — the period between the first and last non-zero sale. How would you define this? Is first/last non-zero sale sufficient, or do you need a more robust method?
- Cross-reference with the sell_prices table: if an item-store pair has no price record for a given week, what does that imply about availability? Do price records align with your active window estimates?
- How many item-store series have long interior gaps (e.g., >30 consecutive zeros flanked by non-zero sales on both sides)? Are these stockouts, seasonal items, or data issues?
- Compute the "active zero rate" — the proportion of zeros only within each series' active window. How does this differ from the raw zero rate you computed in 3.1?

**Why this matters for everything downstream:**

- **Lag features:** If you compute lag-7 and the item was out of stock 7 days ago, you feed a misleading zero into your model. How would you handle this?
- **Rolling statistics:** A 28-day rolling mean that includes a 2-week stockout period will underestimate the true demand level. How would you handle this?
- **ADI/CV2 classification (section 3.2):** If you include availability-driven zeros in your ADI calculation, you will overstate intermittency. How much does the classification change when you restrict to the active window?
- **Training data:** Should you train on the full history from d_1, or only on the active window? What are the tradeoffs?
- **Cross-validation:** If your CV fold boundary falls in the middle of a stockout period, your evaluation is contaminated. How would you detect and handle this?

**Practical exercise:** Pick 10–15 item-store series with high zero rates. For each one, manually inspect the sales time series alongside the price data. Can you visually distinguish "the item wasn't available" from "the item was available but didn't sell"? What heuristics would you use to automate this distinction at scale?

**Connection to CVS:** At CVS, you likely have perpetual inventory data or on-hand flags that tell you directly whether an item was in stock. M5 does not give you this. But the techniques you develop here — using price data as a proxy for availability, detecting store closure days from aggregate patterns, defining active windows — are exactly the same techniques you need when your inventory signal is noisy or delayed, which happens often in practice even when the data nominally exists.

### 3.2 Sparsity and Demand Classification (~3 hours)

This is the most important exploration task. Compute for every item-store series:

- **Proportion of zeros** (% of days with zero sales)
- **ADI** (Average Demand Interval) — average number of days between non-zero demands
- **CV-squared of non-zero demand** — coefficient of variation squared of the demand sizes when demand does occur

Use the ADI/CV2 framework from Boylan & Syntetos to classify each series:

| | CV2 < 0.49 | CV2 >= 0.49 |
|---|---|---|
| ADI < 1.32 | Smooth | Erratic |
| ADI >= 1.32 | Intermittent | Lumpy |

Plot the distribution. What fraction of item-store series falls into each quadrant? How does this change at higher aggregation levels (item-state, department-store, etc.)?

Why this matters: smooth demand can be forecast with standard methods (ETS, ARIMA). Intermittent and lumpy demand need specialized methods (Croston, TSB). The distribution across these quadrants is a fundamental constraint on your modeling strategy. Figure out what that constraint looks like for M5.

### 3.3 Aggregation and Signal-to-Noise (~3 hours)

At each hierarchy level, compute:

- **Mean daily sales** — does it get large enough to be meaningful?
- **Coefficient of variation (CV)** — ratio of standard deviation to mean. High CV = noisy signal.
- **Proportion of zeros** — does sparsity disappear when you aggregate?

Create a table like this and fill it in yourself:

| Level | # Series | Mean Sales | Median CV | % Zero Days |
|-------|----------|------------|-----------|-------------|
| Item-Store | ? | ? | ? | ? |
| Item-State | ? | ? | ? | ? |
| Dept-Store | ? | ? | ? | ? |
| Category-State | ? | ? | ? | ? |
| Total | ? | ? | ? | ? |

Add more hierarchy levels if you want. The point is to find the level where signal emerges from noise. This table is one of the most important artifacts you will produce in this entire project.

### 3.4 Temporal Patterns (~3 hours)

Pick a representative sample of series from each demand class (smooth, erratic, intermittent, lumpy) and plot:

- **Raw daily sales** over the full history
- **Weekly aggregated sales** — does the noise smooth out?
- **Monthly aggregated sales** — can you see trend or seasonality now?
- **Day-of-week averages** — is there a weekday/weekend pattern?
- **Month-of-year averages** — is there annual seasonality?

Also look at aggregate levels:

- Total sales over time — is there a visible trend? Structural breaks?
- Department-level sales — which departments are growing, shrinking, stable?
- State-level sales — are the states similar or different in their patterns?

### 3.5 Calendar and Event Effects (~2 hours)

- Plot total daily sales overlaid with event markers. Do events create visible spikes or dips?
- Compare SNAP vs. non-SNAP days for the FOODS category in each state. How big is the lift?
- Look at major holidays and recurring events. Which ones affect which departments?
- Are there any obvious structural breaks (e.g., a store opening/closing, a category being added)?

### 3.6 Price Exploration (~2 hours)

- How often do prices change? What is the distribution of weeks-between-price-changes?
- When prices drop, does demand increase? Pick 5–10 items and plot price vs. sales side-by-side.
- Are there items that stay on the same price for months vs. items with constant price changes?
- Do all stores charge the same for a given item, or is there meaningful cross-store price variation?

### 3.7 Initial Baseline Check (~1 hour)

Before you finish Phase 0, fit a single naive model to confirm your understanding:

- Compute seasonal naive forecast (same day last week) for all bottom-level series for the last 28 days of the validation set
- Compute the error (just RMSE or MAE for now, not the full WRMSSE)
- Look at where the naive model fails badly vs. where it does well — what characterizes those series?

This gives you a sanity check: you know what "doing nothing" looks like, and every model you build later must beat this.

---

## Part 4: Tools

### 4.1 Python Libraries for Phase 0

| Library | Purpose | Install |
|---------|---------|---------|
| `pandas` | Data loading, manipulation, aggregation | `pip install pandas` |
| `numpy` | Numerical operations | `pip install numpy` |
| `matplotlib` | Basic plotting, diagnostic charts | `pip install matplotlib` |
| `seaborn` | Statistical visualizations, heatmaps | `pip install seaborn` |
| `plotly` | Interactive exploration (optional but useful for drilling into series) | `pip install plotly` |
| `polars` | Faster alternative to pandas for large operations (optional) | `pip install polars` |
| `jupyter` / `jupyterlab` | Notebook environment | `pip install jupyterlab` |

### 4.2 Data Loading Tips

The M5 sales data is in wide format (days as columns). For most analysis, you will want to melt it to long format:

```python
import pandas as pd

sales = pd.read_csv('data/sales_train_evaluation.csv')
id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
sales_long = sales.melt(id_vars=id_cols, var_name='d', value_name='sales')
sales_long['d_num'] = sales_long['d'].str.extract(r'(\d+)').astype(int)
```

This will produce a large number of rows. If performance is an issue, consider using polars or working with a subset of stores first.

Merging with calendar:
```python
calendar = pd.read_csv('data/calendar.csv')
sales_long = sales_long.merge(calendar, left_on='d', right_on='d', how='left')
```

Merging with prices (match on store_id, item_id, wm_yr_wk):
```python
prices = pd.read_csv('data/sell_prices.csv')
sales_long = sales_long.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
```

### 4.3 Visualization Priorities

In Phase 0, optimize for diagnostic value, not aesthetics. The chart types that matter most:

- **Heatmaps** (seaborn `heatmap`): sales by (item, week) to spot structural patterns, missing data, and item introductions
- **Histograms**: distribution of daily sales, proportion of zeros, CV values across series
- **Line plots**: raw and aggregated time series at various hierarchy levels
- **Box plots**: sales distribution by day-of-week, by month, by SNAP status
- **Scatter plots**: ADI vs. CV2 for demand classification

### 4.4 Useful Terminal Tools

| Tool | Purpose |
|------|---------|
| `wc -l data/*.csv` | Quick row counts |
| `head -n 5 data/calendar.csv` | Preview file structure |
| `csvstat` (from csvkit: `pip install csvkit`) | Quick summary statistics from the command line |
| `git init` | Initialize the repo once you are ready (Phase 2) |

---

## Part 5: Exploration Log Template

The file `docs/exploration_log.md` has been created. Use this format for each entry:

```markdown
## 2026-04-XX — [Topic]

**Question:** What was I trying to find out?

**What I did:** Brief description of the analysis.

**What I found:** Key numbers, observations, surprises.

**Hypothesis:** What does this imply for modeling?

**Next:** What should I look at next based on this?
```

Date every entry. Be specific with numbers (e.g., "X% of item-store series have >50% zeros" is useful; "the data is sparse" is not). Include notebook cell references so you can find the code later.

---

## Part 6: Checklist — Phase 0 Complete When You Can Answer

- [ ] How many item-store series are there, and what fraction are intermittent?
- [ ] Can you distinguish true zero demand from missing/unavailable data? What heuristic did you use, and how much does it change your sparsity numbers?
- [ ] At what aggregation level does the coefficient of variation drop below 1.0 for most series?
- [ ] What are the top 3 strongest temporal patterns in this data?
- [ ] Which departments behave most differently from each other?
- [ ] What does the price data tell you — are most items stable-price or promotion-driven?
- [ ] What does seasonal naive achieve as a baseline, and where does it fail?
- [ ] Based on all of the above, what aggregation level and time granularity would you start modeling at, and why?

The last question is the most important output of Phase 0. Your answer becomes ADR-001 in Phase 2.

---

## Appendix: Validation Reference

Use this appendix **after** you have completed your own exploration to check whether your findings are in the right ballpark. Do not read ahead — the value of Phase 0 is in discovering these things yourself.

### A.1 Data Shape

- Confirm your count of unique items, stores, departments, categories, and states against the ID columns in the sales file. The sales file has one row per item-store combination.
- The calendar file maps each `d_` column to an actual date. Verify there are no gaps by checking that the date column increments by exactly one day throughout.
- To detect items introduced mid-history: look for series where the first non-zero sale occurs well after d_1. A leading run of zeros longer than, say, 90 days is a strong signal of a later introduction. Cross-reference with the sell_prices table — if an item-store pair has no price record before a certain week, it likely was not on the shelf yet.
- For the zero-sales ratio: compute `(sales == 0).sum() / len(sales)` across all item-store-day observations. You should find that zeros are the majority — by a lot. If your number is below 50%, double-check your calculation.

### A.1b Data Continuity and Zero vs. Missing

- **Calendar gaps:** Check whether the date column in `calendar.csv` has any missing dates by converting to datetime and computing the difference between consecutive rows. Every gap should be exactly 1 day. If you find gaps, they matter. If you don't, good — but that only means the calendar is continuous, not that every series has data for every day.
- **Store closure detection:** Compute total sales per store per day. Days where a store's total across all items drops to zero (or implausibly close to zero) are likely closure days. Look at whether these cluster on specific dates (holidays) or appear randomly. If a store shows zero total sales on a day that is not a plausible closure day, that may be a data issue worth flagging.
- **Active window via prices:** For each item-store, find the first and last `wm_yr_wk` that appears in the `sell_prices` table. Convert these to date ranges using the calendar. Compare this to the first and last non-zero sale in the sales data. They should be roughly consistent — the price record should start on or before the first sale and end on or after the last sale. If the price record starts significantly before the first sale, the item may have been on shelf but not selling (true intermittent demand). If the first sale comes before the first price record, there may be a data join issue.
- **Active zero rate vs. raw zero rate:** When you compute the proportion of zeros only within each series' active window (defined by price availability), the number should be lower than the raw zero rate from section 3.1. How much lower tells you how much of the apparent sparsity is actually a data availability artifact rather than true intermittent demand. If the two rates are similar, availability-driven zeros are not a major factor. If they differ substantially, your entire sparsity analysis (and downstream modeling) needs to account for this.
- **Interior gaps:** For series with long runs of consecutive zeros in the middle of their active window, check whether the price data also disappears during those gaps. If prices are missing too, the item was likely pulled from the shelf (temporary delisting, seasonal item, etc.). If prices are present but sales are zero for weeks, that's more likely a genuine slow-moving or stockout pattern.
- **Impact on ADI/CV2:** Re-run your demand classification from section 3.2 using only the active window for each series. Compare the quadrant distribution to the version that uses the full history. The shift tells you how much your demand taxonomy depends on getting the availability definition right — which is a critical sensitivity for production systems.

### A.2 Sparsity and Demand Classification

- When you run the ADI/CV2 classification, expect the majority of item-store series to land outside the "smooth" quadrant. If you find that most series are classified as smooth, recheck your ADI calculation — remember ADI is computed only over the period where the item was actually available (not from d_1 if the item was introduced later).
- At the department-store level, most series should shift toward the smooth/erratic side of the framework. If they don't, check that your aggregation is summing across all items within the department-store group.
- The HOBBIES category will behave differently from FOODS. FOODS has higher volume and lower intermittency. HOBBIES has lower volume and higher intermittency. If your classification doesn't show this distinction, investigate.

### A.3 Aggregation and Signal-to-Noise

- CV should decrease monotonically as you aggregate from item-store to higher levels. If it doesn't at some level, that's worth investigating — it may indicate that you're mixing very different demand patterns within a group.
- The zero-day proportion should approach zero well before you reach the total level. If it's still meaningfully above zero at the department-store level, check your aggregation logic.
- Compare your mean daily sales at the item-store level to the total level. The ratio should roughly equal the number of item-store series (since total = sum of all item-stores). This is a basic sanity check on your aggregation.

### A.4 Temporal Patterns

- Day-of-week: compute the mean sales for each weekday across all series. If Saturday and Sunday don't stand out from weekdays, check whether you joined the calendar data correctly.
- Annual seasonality: plot monthly averages across years. Look particularly at the period around late November through December. If you don't see a pattern there, you may be averaging across too heterogeneous a set of items.
- Trend: at the total level, plot yearly sales (sum of all daily sales per year). Is the total stable, growing, or has any structural shift? Compare your conclusion to what you see at the state level — the states may not all have the same trend.

### A.5 Calendar and Event Effects

- SNAP effect: for each state, separate the FOODS category sales into SNAP days and non-SNAP days for the same month. Compute the mean daily sales for each group. If there's a SNAP effect, the SNAP-day mean should be visibly higher. The magnitude will differ by state because the SNAP schedules differ.
- For holidays: Christmas and Thanksgiving should show a distinct pattern. But be careful — some holidays cause a spike *before* the actual day (pre-holiday shopping) and a dip *on* the day (stores closed or reduced hours). Check whether you see both sides of this pattern.
- Structural breaks: look at each store's total sales over time. A sudden, permanent level shift could indicate a remodel, a new competitor, or a change in the data collection. If you see one, note the date and the store — it may affect how you set up training windows later.

### A.6 Price Exploration

- Compute the number of distinct prices per item-store over the full history. Some items will have very few distinct prices (staples), others will have many (promotion-heavy). If you histogram this distribution, it should be right-skewed — most items have a modest number of price points, but a tail of heavily promoted items have many.
- To check for price elasticity: for items with frequent price changes, compute the correlation between (week-over-week price change) and (week-over-week sales change). A negative correlation (price goes down, sales go up) is the expected sign for elastic demand. If you find items where this relationship is absent or reversed, those may be inelastic staples or items where the price change is too small to matter.
- Cross-store price variation: for a given item and week, check how many different prices exist across the stores that carry that item. If there is no cross-store variation, then store-level price features add no information. If there is, that's a useful feature for modeling.

### A.7 Naive Baseline

- Your seasonal naive forecast (predict each day's sales as the sales from the same weekday one week ago) should produce an RMSE that varies enormously across series. High-volume smooth series will have moderate relative error. Intermittent series will often have RMSE close to the mean demand itself — the naive forecast for a zero day is whatever happened 7 days ago, which may or may not be zero.
- As a rough sanity check: compute the overall MAE across all series. Then compute the mean daily sales across all series. The ratio (MAE / mean) gives you a crude sense of baseline accuracy. If this ratio is below 0.5, something may be wrong with your forecast alignment (make sure your forecast dates match your actual dates). If it's above 3.0, check that you haven't accidentally forecast on a period with a structural break.
- The series where naive performs worst are the ones where real models have the most room to add value. Characterize them: are they intermittent? Event-sensitive? Recently introduced? This characterization feeds directly into your modeling strategy in later phases.
