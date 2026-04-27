# Exploration Log

<!-- Date every entry. Be specific with numbers. Reference notebook cells so you can trace back. -->

---

## 2026-04-27

**Dataset structure**
- 10 stores across 3 states: CA (4 stores), TX (3), WI (3)
- 3 categories per store: FOODS (1,437 SKUs), HOBBIES (565), HOUSEHOLD (1,047) — 3,049 total SKUs
- 30,490 store-item combinations total

**Sell prices (`sell_prices.csv`)**
- Granularity: store × item × Walmart week (`wm_yr_wk`)
- `wm_yr_wk` format: `1` + 2-digit year + 2-digit week (e.g. 11101 = 2011 week 1, starts Saturdays)
- At a high level, prices do fluctuate — same product can show different price change patterns across stores
- **8,247 store-item pairs have a completely fixed price (27% of 30,490)** — worth keeping in mind for feature engineering; a "price change" feature will be zero for these by definition

---
