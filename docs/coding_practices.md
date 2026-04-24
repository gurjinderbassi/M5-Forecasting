# Coding Practices — Building Professional Muscle Memory

**The principle:** Your notebook is a lab report. It calls functions, displays results, and records interpretations. It does not contain logic. The moment you write a `for` loop or an `if` statement that you might use again, it goes into a function. The moment you write a function you might use in another notebook, it goes into a module.

This is not about aesthetics. It is about the fact that when you copy-paste a block of code and tweak one parameter, you now have two places to fix when you find a bug. At CVS scale with dozens of analyses running, that kills you.

---

## The Three-Layer Pattern

Every analysis you do should have three layers. This applies even in Phase 0 when you don't have a clean `src/` directory yet.

```
Layer 1: Module (.py file)     — Pure functions. No side effects. No display. No hardcoded paths.
Layer 2: Notebook helpers       — Thin wrappers that call Layer 1 and format for display.
Layer 3: Notebook cells         — Call helpers. Look at output. Write interpretation.
```

Layer 1 is the thing you keep forever and reuse across notebooks. Layer 2 is convenience glue. Layer 3 is disposable narration.

---

## Concrete Example: Decomposition Across Granularity Levels

You want to study whether signal emerges from noise as you move up the hierarchy. Here is how to do this without repeating yourself.

### Step 1: Define what you need before writing any code

Before you touch a keyboard, write down (in the notebook markdown cell or on paper):

- **Input:** A time series (a pandas Series with a datetime index and numeric values).
- **Operation:** Decompose it (STL or classical), extract trend, seasonal, residual.
- **Output:** The decomposition components, plus summary stats (strength of trend, strength of seasonality).
- **Variation axis:** I want to do this at item-store, item-state, dept-store, category-state, and total levels.

The variation axis is the key. Whatever varies across your analyses is the **parameter**. Whatever stays the same is the **function body**. If you find yourself about to copy a cell and change one thing, stop — that one thing is a parameter.

### Step 2: Write the atomic function first

Start a file. Even in Phase 0, create a file called `helpers.py` in your project root or in `notebooks/`. You will migrate it to `src/` later.

```python
# helpers.py

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL


def decompose_series(series: pd.Series, period: int = 7) -> dict:
    """
    STL decomposition of a single time series.
    
    Parameters
    ----------
    series : pd.Series
        Time series with a datetime-like index. Must not contain NaNs.
    period : int
        Seasonal period (7 for daily data with weekly seasonality).
    
    Returns
    -------
    dict with keys: 'trend', 'seasonal', 'resid', 'strength_of_trend',
    'strength_of_seasonality', 'series_name'
    """
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    
    # Strength of trend and seasonality (Hyndman & Athanasopoulos, FPP3 Ch 3)
    resid_var = np.var(result.resid)
    strength_trend = max(0, 1 - resid_var / np.var(result.trend + result.resid))
    strength_seasonal = max(0, 1 - resid_var / np.var(result.seasonal + result.resid))
    
    return {
        'trend': result.trend,
        'seasonal': result.seasonal,
        'resid': result.resid,
        'strength_of_trend': strength_trend,
        'strength_of_seasonality': strength_seasonal,
        'series_name': series.name,
    }
```

Notice what this function does NOT do:

- It does not load data.
- It does not know about M5 column names, hierarchy levels, or file paths.
- It does not plot anything.
- It does not print anything.
- It takes a generic pandas Series and returns a generic dict.

This is the muscle memory rule: **a function that computes something should never also display something or load something.** Separate concerns completely.

### Step 3: Write the aggregation function

You need to go from raw M5 data to an aggregated series at any hierarchy level. This is a separate concern from decomposition.

```python
# helpers.py (continued)

def aggregate_sales(
    sales_long: pd.DataFrame,
    group_cols: list[str],
    date_col: str = 'date',
    value_col: str = 'sales',
    freq: str = 'D',
) -> pd.DataFrame:
    """
    Aggregate sales to a given hierarchy level and time frequency.
    
    Parameters
    ----------
    sales_long : pd.DataFrame
        Long-format sales data. Must contain date_col, value_col, and all group_cols.
    group_cols : list[str]
        Columns defining the hierarchy level. 
        E.g., ['dept_id', 'store_id'] for dept-store level.
        Pass [] for total level.
    date_col : str
        Column containing dates.
    value_col : str
        Column containing sales values.
    freq : str
        Resample frequency. 'D' for daily, 'W' for weekly, 'MS' for monthly.
    
    Returns
    -------
    pd.DataFrame with one row per group per period, columns: group_cols + ['date', 'sales']
    """
    if not group_cols:
        # Total level
        agg = (
            sales_long
            .groupby(date_col)[value_col]
            .sum()
            .resample(freq)
            .sum()
            .rename('sales')
            .to_frame()
        )
        agg['_group'] = 'total'
        return agg
    
    agg = (
        sales_long
        .groupby(group_cols + [date_col])[value_col]
        .sum()
        .reset_index()
    )
    agg[date_col] = pd.to_datetime(agg[date_col])
    agg = agg.set_index(date_col)
    
    # Resample within each group
    resampled = (
        agg
        .groupby(group_cols)[value_col]
        .resample(freq)
        .sum()
        .reset_index()
    )
    return resampled
```

Again: no plotting, no printing, no M5-specific hardcoding. The group_cols parameter is what lets you reuse this for any hierarchy level without copy-pasting.

### Step 4: Write the "sweep" function

This is the function that does the same analysis across many levels or many series. This is where the reuse payoff happens.

```python
# helpers.py (continued)

def decomposition_summary_by_level(
    sales_long: pd.DataFrame,
    hierarchy_levels: dict[str, list[str]],
    period: int = 7,
    freq: str = 'D',
    sample_per_level: int | None = None,
) -> pd.DataFrame:
    """
    Run STL decomposition across multiple hierarchy levels and return
    a summary table of trend/seasonality strength.
    
    Parameters
    ----------
    sales_long : pd.DataFrame
        Long-format sales data with a 'date' column.
    hierarchy_levels : dict
        Keys are level names, values are lists of grouping columns.
        E.g., {'item-store': ['item_id', 'store_id'], 'total': []}
    period : int
        Seasonal period for STL.
    freq : str
        Time frequency to aggregate to before decomposing.
    sample_per_level : int or None
        If set, randomly sample this many series per level instead of
        decomposing all of them (useful for large levels).
    
    Returns
    -------
    pd.DataFrame with columns: level, group, strength_of_trend,
    strength_of_seasonality
    """
    rows = []
    
    for level_name, group_cols in hierarchy_levels.items():
        agg = aggregate_sales(sales_long, group_cols, freq=freq)
        
        if group_cols:
            groups = agg.groupby(group_cols)
            group_keys = list(groups.groups.keys())
            
            if sample_per_level and len(group_keys) > sample_per_level:
                rng = np.random.default_rng(42)
                group_keys = list(rng.choice(group_keys, sample_per_level, replace=False))
            
            for key in group_keys:
                subset = groups.get_group(key)
                series = subset.set_index('date')['sales']
                series.name = str(key)
                
                if len(series.dropna()) < 2 * period:
                    continue
                
                try:
                    result = decompose_series(series, period=period)
                    rows.append({
                        'level': level_name,
                        'group': str(key),
                        'strength_of_trend': result['strength_of_trend'],
                        'strength_of_seasonality': result['strength_of_seasonality'],
                    })
                except Exception:
                    continue
        else:
            series = agg['sales']
            series.name = 'total'
            result = decompose_series(series, period=period)
            rows.append({
                'level': level_name,
                'group': 'total',
                'strength_of_trend': result['strength_of_trend'],
                'strength_of_seasonality': result['strength_of_seasonality'],
            })
    
    return pd.DataFrame(rows)
```

The key design: `hierarchy_levels` is a dict that maps human-readable level names to the columns that define that level. You define it once in your notebook, and the function sweeps across all of them. When you want to add a new level, you add one entry to the dict — you never touch the function.

### Step 5: The notebook — thin, interpretive, no logic

```python
# Cell 1 — Imports and data loading (the only place paths appear)
import pandas as pd
from helpers import aggregate_sales, decompose_series, decomposition_summary_by_level

sales_long = pd.read_parquet('data/sales_long.parquet')  # or however you prepared it
```

```python
# Cell 2 — Define hierarchy levels (configuration, not logic)
HIERARCHY_LEVELS = {
    'total':          [],
    'state':          ['state_id'],
    'category':       ['cat_id'],
    'dept-store':     ['dept_id', 'store_id'],
    'item-state':     ['item_id', 'state_id'],
    'item-store':     ['item_id', 'store_id'],
}
```

```python
# Cell 3 — Run the sweep (one line)
summary = decomposition_summary_by_level(
    sales_long,
    HIERARCHY_LEVELS,
    period=7,
    freq='W',
    sample_per_level=200,
)
summary.head(20)
```

```python
# Cell 4 — Visualize (plotting is a notebook concern, not a helpers concern)
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(data=summary, x='level', y='strength_of_trend', ax=axes[0])
axes[0].set_title('Strength of Trend by Hierarchy Level')
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(data=summary, x='level', y='strength_of_seasonality', ax=axes[1])
axes[1].set_title('Strength of Seasonality by Hierarchy Level')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
```

```markdown
# Cell 5 — Interpretation (markdown cell)

## Findings

[Write your observations here. Which levels show strong trend? 
Where does seasonality peak? At what level is the residual 
dominant — i.e., the signal is mostly noise?]

## Implications

[What does this mean for your modeling strategy?]
```

Count the lines of actual code in the notebook: about 20, plus markdown. All logic lives in `helpers.py`. If you find a bug in your STL calculation, you fix it in one place and re-run. If your colleague wants to add a `store x category` level, they add one line to the dict.

---

## The Rules (Tattoo These On Your Brain)

### Rule 1: The Parameter Test

Before copy-pasting a cell, ask: "What am I about to change?" That thing is a parameter. Extract a function with that parameter.

Bad:
```python
# Decomposition for total level
total_series = sales_long.groupby('date')['sales'].sum()
stl = STL(total_series, period=7)
result = stl.fit()
# ... 10 lines of processing ...

# Decomposition for state level   <-- copy-pasted and tweaked
ca_series = sales_long[sales_long.state_id == 'CA'].groupby('date')['sales'].sum()
stl = STL(ca_series, period=7)
result = stl.fit()
# ... same 10 lines ...
```

Good:
```python
total_series = aggregate_sales(sales_long, [])['sales']
ca_series = aggregate_sales(sales_long[sales_long.state_id == 'CA'], [])['sales']

total_decomp = decompose_series(total_series)
ca_decomp = decompose_series(ca_series)
```

Better:
```python
summary = decomposition_summary_by_level(sales_long, HIERARCHY_LEVELS)
```

### Rule 2: Functions Compute, Notebooks Display

A function should return data. A notebook cell should display it. Never mix these. If you put `plt.show()` inside a function, you cannot reuse that function in a pipeline, a test, or a different notebook with a different layout.

Exception: you can write plotting helper functions, but they should be clearly labeled as such and kept separate from computation functions.

```python
# This is fine — a dedicated plotting function
def plot_decomposition(decomp_result: dict, figsize=(14, 8)):
    """Plot STL decomposition components."""
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    for ax, key, title in zip(
        axes,
        ['trend', 'seasonal', 'resid'],
        ['Trend', 'Seasonal', 'Residual'],
    ):
        ax.plot(decomp_result[key])
        ax.set_title(title)
    axes[0].plot(... original series ...)  # you get the idea
    plt.tight_layout()
    return fig  # return, don't show — let the notebook decide
```

### Rule 3: Configuration Over Code

Anything that defines "what" you are analyzing (which levels, which period, which sample size) should be in a dict, a YAML file, or a constant at the top of the notebook — never buried inside function calls deep in a cell.

```python
# Top of notebook — all knobs in one place
HIERARCHY_LEVELS = { ... }
STL_PERIOD = 7
TIME_FREQ = 'W'
SAMPLE_SIZE = 200
```

This is the precursor to the config-driven experiment management you will build in Phase 2. Start the habit now.

### Rule 4: Name Things By What They Represent, Not How They Were Made

Bad: `df2`, `result_agg`, `temp`, `merged_df`
Good: `sales_weekly_by_dept`, `decomp_summary`, `active_window_mask`, `price_change_frequency`

Bad: `do_analysis(df, 'CA')` 
Good: `decomposition_summary_by_level(sales_long, {'state-CA': ['state_id']})`

If you cannot name a variable in a way that another person would understand without seeing the code that created it, refactor until you can.

### Rule 5: Type Hints and Docstrings on Everything in Layer 1

Not because it's pretty. Because when you come back in three weeks, or when you're building Phase 4 on top of Phase 0 code, you need to know what goes in and what comes out without reading the implementation.

```python
def decompose_series(series: pd.Series, period: int = 7) -> dict:
    """
    STL decomposition of a single time series.
    
    Parameters
    ----------
    series : pd.Series
        Time series with a datetime-like index. Must not contain NaNs.
    period : int
        Seasonal period (7 for daily data with weekly seasonality).
    
    Returns
    -------
    dict with keys: 'trend', 'seasonal', 'resid', 'strength_of_trend',
    'strength_of_seasonality', 'series_name'
    """
```

The discipline: write the docstring BEFORE the implementation. It forces you to think about the interface before the logic. This is design-by-contract.

### Rule 6: Early Returns and Defensive Checks

Production data is messy. Functions should fail informatively, not silently produce garbage.

```python
def decompose_series(series: pd.Series, period: int = 7) -> dict:
    if series.isna().any():
        raise ValueError(f"Series '{series.name}' contains NaNs. Handle before decomposing.")
    if len(series) < 2 * period:
        raise ValueError(
            f"Series '{series.name}' has {len(series)} observations, "
            f"need at least {2 * period} for period={period}."
        )
    # ... proceed with actual logic
```

At CVS scale, you will encounter every possible data pathology. Building the habit of checking inputs early saves hours of debugging silent numerical errors later.

---

## Notebook Naming and Structure

```
notebooks/
├── 00_data_loading.ipynb          # Load, melt, save to parquet. Run once.
├── 01_shape_and_completeness.ipynb
├── 02_continuity_and_availability.ipynb
├── 03_sparsity_classification.ipynb
├── 04_aggregation_signal_noise.ipynb
├── 05_decomposition_by_level.ipynb
├── 06_temporal_patterns.ipynb
├── 07_calendar_events.ipynb
├── 08_price_exploration.ipynb
├── 09_naive_baseline.ipynb
└── helpers.py                     # Shared functions. Migrates to src/ in Phase 2.
```

Every notebook starts with the same structure:

```python
# Cell 1: Purpose (markdown)
"""
# 05 — Decomposition by Hierarchy Level
 
**Question:** At which aggregation level does trend/seasonality become 
detectable above noise?

**Approach:** STL decomposition at multiple hierarchy levels, compare 
strength-of-trend and strength-of-seasonality distributions.

**Dependencies:** helpers.py, data/sales_long.parquet (from notebook 00)
"""
```

```python
# Cell 2: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import (
    aggregate_sales,
    decompose_series, 
    decomposition_summary_by_level,
)
```

```python
# Cell 3: Configuration
HIERARCHY_LEVELS = { ... }
STL_PERIOD = 7
# ... etc
```

```python
# Cell 4: Load data (one line, path to prepared data, not raw CSVs)
sales_long = pd.read_parquet('data/sales_long.parquet')
```

```python
# Cells 5+: Analysis — function calls, display, markdown interpretation
```

```python
# Final cell: Summary (markdown)
"""
## Key Findings
- ...

## Logged to exploration_log.md: Yes / No
"""
```

---

## The helpers.py Growth Pattern

In Phase 0, `helpers.py` starts small and grows organically. Don't try to design it upfront. Here is the natural progression:

**Week 1:** You write a function inline in a notebook. You use it once.

**Still Week 1:** You need it again in the next cell with different parameters. You realize it should be a function. You move it to the top of the notebook.

**Day 3:** You need it in a different notebook. You move it to `helpers.py` and import it. THIS is the trigger for extracting to a module — the second notebook that needs it.

**Week 2:** `helpers.py` has 15 functions and is getting unwieldy. You split it:

```
notebooks/
├── helpers/
│   ├── __init__.py
│   ├── loading.py        # Data loading and reshaping
│   ├── aggregation.py    # Hierarchy aggregation
│   ├── decomposition.py  # STL and related
│   ├── classification.py # ADI/CV2 demand classification
│   └── plotting.py       # Visualization helpers
```

**Phase 2:** You migrate `helpers/` to `src/` and add tests. The functions are already clean because you wrote them properly from Day 1.

The key: you never design this structure upfront. You let it emerge from actual reuse patterns. But you follow the rules above so that when extraction time comes, the functions are already clean, typed, documented, and free of side effects.

---

## Anti-Patterns to Catch Yourself Doing

**The "just this once" copy-paste.** There is no such thing. If you copy a cell and change one thing, extract a function immediately. The cost is 2 minutes now. The cost of not doing it is 20 minutes of debugging a stale copy later.

**The mega-cell.** If a notebook cell is more than 15–20 lines, it is doing too much. Break it into a function call and a display step.

**The magic number.** `sales_long[sales_long.d_num > 500]` — what is 500? Why 500? Use a named constant: `FIRST_STABLE_DAY = 500  # skip initial period with incomplete item coverage`.

**The unnamed intermediate DataFrame.** `df2 = df.merge(...)` then `df3 = df2.groupby(...)` — name them by what they contain: `sales_with_prices = sales_long.merge(...)`, `weekly_dept_sales = sales_with_prices.groupby(...)`.

**Importing everything.** `from helpers import *` makes it impossible to know where a function came from. Always import explicitly.

**Plotting inside computation loops.** If your sweep function generates 200 plots as a side effect, you can't run it headless, can't test it, and can't use it in a pipeline. Compute first, collect results, plot separately.

---

## A Note on When to Break the Rules

Phase 0 is exploration. You WILL write throwaway code. That's fine. The rules above apply to any code that:

- You use more than once
- Produces a number or chart that goes into your exploration log
- Might be useful in a later phase

For a quick one-off "let me just check something" cell, write it inline, look at the result, and either delete the cell or promote the code to a function if it turned out to be useful. The judgment of "is this throwaway or reusable" is itself a skill you're building.
