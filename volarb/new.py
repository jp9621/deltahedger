import datetime
import math
import re

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from polygon import RESTClient

# ---- BLACK–SCHOLES IMPL. VOL. (bisection) ----
def implied_vol(S, K, T, r, price, tol=1e-6, max_iter=60):
    low, high = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        d1  = (math.log(S / K) + (r + 0.5 * mid * mid) * T) / (mid * math.sqrt(T))
        d2  = d1 - mid * math.sqrt(T)
        bs  = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        if bs > price:
            high = mid
        else:
            low  = mid
    return 0.5 * (low + high)

# ---- USER INPUT ----
API_KEY       = 'w2chIPf4EUplQqQv6b8Nxnn8GQV8pfGC'
option_symbol = 'O:AAPL250530C00195000'  # e.g. 'O:<SYMBOL><YYMMDD><C/P><strike*1000>'

# ---- PARSE UNDERLYING (chars until first digit) ----
first_digit_idx = re.search(r'\d', option_symbol).start()
underlying_sym  = option_symbol[2:first_digit_idx]

# ---- PARSE EXPIRATION DATE (YYMMDD) ----
exp_str         = option_symbol[first_digit_idx:first_digit_idx+6]
yy, mm, dd      = int(exp_str[:2]), int(exp_str[2:4]), int(exp_str[4:6])
expiration_date = datetime.datetime(2000 + yy, mm, dd)

# ---- PARSE OPTION TYPE (C or P) ----
opt_type_char = option_symbol[first_digit_idx+6]
option_type   = 'call' if opt_type_char == 'C' else 'put'

# ---- PARSE STRIKE (last 8 digits → float dollars) ----
strike_str = option_symbol[first_digit_idx+7:]
K          = int(strike_str) / 1000.0

# ---- REMAINING INPUTS ----
date_str     = '2025-05-23'   # trading day to process (YYYY-MM-DD)
r            = 0.01           # annual risk-free rate

# ---- HELPER: convert "YYYY-MM-DD" → market open/close in ms ----
def get_day_bounds_unix_ms(date_str):
    date         = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    market_open  = date.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = date.replace(hour=16, minute=0,  second=0, microsecond=0)
    return int(market_open.timestamp() * 1000), int(market_close.timestamp() * 1000)

start_ms, end_ms   = get_day_bounds_unix_ms(date_str)
expiration_ms      = int(expiration_date.timestamp() * 1000)

# ---- INSTANTIATE POLYGON CLIENT ----
client = RESTClient(api_key=API_KEY)

# ---- FETCH 1-MIN AGGREGATES VIA RESTClient ----
opt_resp = client.get_aggs(
    ticker=option_symbol,
    multiplier=1,
    timespan="minute",
    from_=start_ms,
    to=end_ms,
    adjusted=True,
    sort="asc",
    limit=50000,
)

stk_resp = client.get_aggs(
    ticker=underlying_sym,
    multiplier=1,
    timespan="minute",
    from_=start_ms,
    to=end_ms,
    adjusted=True,
    sort="asc",
    limit=50000,
)

opt_df = pd.DataFrame(opt_resp)
stk_df = pd.DataFrame(stk_resp)

# ---- TURN MS → DATETIME INDEX ----
opt_df['dt'] = pd.to_datetime(opt_df['timestamp'], unit='ms')
stk_df['dt'] = pd.to_datetime(stk_df['timestamp'], unit='ms')

# ---- MERGE ON EXACT MINUTE STAMPS ----
merged = pd.merge(
    opt_df,
    stk_df[['dt', 'close']].rename(columns={'close': 'stk_close'}),
    on='dt',
    how='inner'
)
print(f"Found {len(merged)} matching minute‐bars out of "
      f"{len(opt_df)} option / {len(stk_df)} underlying rows")

# ---- COMPUTE IV FOR EACH BAR ----
ivs = []
for row in merged.itertuples(index=False):
    S     = float(row.stk_close)    # underlying close
    price = float(row.close)        # option close
    t_ms  = int(row.dt.value // 10**6)
    T     = max((expiration_ms - t_ms) / 1000.0 / (365 * 24 * 3600), 1e-6)
    iv    = implied_vol(S, K, T, r, price)
    ivs.append(iv)

merged['iv'] = ivs

# ---- COMPUTE 10-BAR ROLLING AVERAGE OF IMPLIED VOL ----
merged['iv_rolling10'] = merged['iv'].rolling(window=3, min_periods=1).mean()

# ---- COMPUTE SPREAD AND EFFICIENCY ----
merged['spread'] = merged['close'] - merged['open']
merged['efficiency'] = merged['spread'] / merged['volume'].replace(0, pd.NA)

# ---- === PLOTTING === ----

# (1) Plot: 10-Bar Rolling Implied Volatility (standalone)
fig_iv, ax_iv = plt.subplots(figsize=(10, 4))
ax_iv.plot(merged['dt'], merged['iv_rolling10'], label='10-Bar Rolling IV', color='blue')
ax_iv.set_xlabel('Time')
ax_iv.set_ylabel('Rolling IV')
ax_iv.set_title(f'10-Bar Rolling Implied Volatility for {option_symbol} on {date_str}')
ax_iv.legend(loc='best')
fig_iv.tight_layout()
fig_iv.savefig("data/implied_vol_plot.png")


# (2) Plot: Spread and Volume over time (shared x-axis, twin y-axes)
fig_sv, ax_sp = plt.subplots(figsize=(10, 4))

# Left y-axis: Spread
ax_sp.plot(merged['dt'], merged['spread'], label='Spread (Close - Open)', color='purple')
ax_sp.set_xlabel('Time')
ax_sp.set_ylabel('Spread (USD)', color='purple')
ax_sp.tick_params(axis='y', labelcolor='purple')

# Right y-axis: Volume
ax_vol = ax_sp.twinx()
ax_vol.plot(merged['dt'], merged['volume'], label='Volume', color='orange')
ax_vol.set_ylabel('Volume', color='orange')
ax_vol.tick_params(axis='y', labelcolor='orange')

# Combine legends
lines_sp, labels_sp = ax_sp.get_legend_handles_labels()
lines_vol, labels_vol = ax_vol.get_legend_handles_labels()
ax_sp.legend(lines_sp + lines_vol, labels_sp + labels_vol, loc='best')

ax_sp.set_title(f'Spread & Volume for {option_symbol} on {date_str}')
fig_sv.tight_layout()
fig_sv.savefig("data/spread_volume_plot.png")
