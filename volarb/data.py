import datetime, math
from polygon import RESTClient
from scipy.stats import norm
import matplotlib.pyplot as plt
import re
import pandas as pd

# ---- BLACK–SCHOLES IMPL. VOL. (bisection) ----
def implied_vol(S, K, T, r, price, tol=1e-6, max_iter=60):
    low, high = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        d1  = (math.log(S/K) + (r + 0.5*mid*mid)*T) / (mid*math.sqrt(T))
        d2  = d1 - mid*math.sqrt(T)
        bs  = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        if bs > price:
            high = mid
        else:
            low  = mid
    return 0.5*(low + high)

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
date_str     = '2025-05-23'   # trading day to plot (YYYY-MM-DD)
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
#   keep only rows where the option bar and underlying bar align
merged = pd.merge(
    opt_df,
    stk_df[['dt','close']].rename(columns={'close':'stk_close'}),
    on='dt', how='inner'
)
print(f"Found {len(merged)} matching minute‐bars out of " 
      f"{len(opt_df)} option / {len(stk_df)} underlying rows")

# ---- COMPUTE IVs ----
times = []
ivs   = []
for row in merged.itertuples(index=False):
    # S = underlying close, price = option close
    S     = float(row.stk_close)
    price = float(row.close)
    t_ms  = int(row.dt.value // 10**6)
    # time to expiry in years
    T     = max((expiration_ms - t_ms)/1000.0/(365*24*3600), 1e-6)
    iv    = implied_vol(S, K, T, r, price)
    times.append(row.dt)   # use the pandas Timestamp
    ivs.append(iv)

merged['iv'] = ivs
merged.to_csv("data/data0.csv")
# ---- PLOT WITH TWO Y-AXES ----
fig, ax_iv = plt.subplots(figsize=(10,4))

# Plot implied vol on the left axis
ax_iv.plot(times, ivs, label='Implied Volatility', color='blue')
ax_iv.set_xlabel('Time')
ax_iv.set_ylabel('Implied Volatility', color='blue')
ax_iv.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for the stock price
ax_price = ax_iv.twinx()
ax_price.plot(merged['dt'], merged['stk_close'], label='Underlying Close', color='orange')
ax_price.set_ylabel('Underlying Close ($)', color='orange')
ax_price.tick_params(axis='y', labelcolor='orange')

# Combine legends
lines_iv, labels_iv     = ax_iv.get_legend_handles_labels()
lines_price, labels_price = ax_price.get_legend_handles_labels()
ax_iv.legend(lines_iv + lines_price, labels_iv + labels_price, loc='best')

plt.title(f'IV & Underlying Price of {option_symbol} on {date_str}')
plt.tight_layout()
plt.savefig("data/data0fig.png")

