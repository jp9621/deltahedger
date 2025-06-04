import datetime
import math
import re

import pandas as pd
from polygon import RESTClient
from scipy.stats import norm

# Import the OptionHedger class (make sure option_manager.py is present)
from OptionHedger import OptionHedger


# ---- BLACK–SCHOLES IMPLIED VOLATILITY (bisection) ----
def implied_vol(S, K, T, r, price, tol=1e-6, max_iter=60):
    low, high = 1e-6, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        d1 = (math.log(S / K) + (r + 0.5 * mid * mid) * T) / (mid * math.sqrt(T))
        d2 = d1 - mid * math.sqrt(T)
        bs = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        if bs > price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


# ---- USER INPUT ----
API_KEY = 'w2chIPf4EUplQqQv6b8Nxnn8GQV8pfGC'
# Polygon‐format option ticker (for example: AAPL June 2025 195 call)
c_option_symbol = 'O:SPY250630C00600000'
p_option_symbol = 'O:SPY250630P00595000'
date_str = '2025-05-23'   # trading day to fetch minute data
r = 0.01                  # annual risk‐free rate


# ---- PARSE OPTION SYMBOLS ----
# Parse call option
call_data = OptionHedger.parse_option_symbol(c_option_symbol)
# Parse put option
put_data = OptionHedger.parse_option_symbol(p_option_symbol)

# Use call data for underlying symbol since both are on same underlying
underlying_sym = call_data['underlying_sym']

start_ms, end_ms = OptionHedger.get_day_bounds_unix_ms(date_str)
expiration_ms = int(call_data['expiration_date'].timestamp() * 1000)


# ---- INSTANTIATE POLYGON CLIENT ----
client = RESTClient(api_key=API_KEY)


# ---- FETCH 1-MINUTE AGGREGATES FOR OPTIONS & UNDERLYING ----
# Fetch call option data
call_resp = client.get_aggs(
    ticker=c_option_symbol,
    multiplier=1,
    timespan="minute",
    from_=start_ms,
    to=end_ms,
    adjusted=True,
    sort="asc",
    limit=50000,
)

# Fetch put option data
put_resp = client.get_aggs(
    ticker=p_option_symbol,
    multiplier=1,
    timespan="minute",
    from_=start_ms,
    to=end_ms,
    adjusted=True,
    sort="asc",
    limit=50000,
)

# Fetch underlying data
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


# ---- CONVERT TO DATAFRAMES AND PREPARE DATA ----
call_df = pd.DataFrame(call_resp)   # raw call option minute-bars
put_df = pd.DataFrame(put_resp)     # raw put option minute-bars
stk_df = pd.DataFrame(stk_resp)     # raw underlying minute-bars

# Get all unique timestamps across all assets
all_timestamps = pd.Series(
    sorted(set(call_df['timestamp'].tolist() + 
             put_df['timestamp'].tolist() + 
             stk_df['timestamp'].tolist()))
)

# Create DataFrames indexed by timestamp for each asset
call_prices = pd.DataFrame(index=all_timestamps)
put_prices = pd.DataFrame(index=all_timestamps)
stk_prices = pd.DataFrame(index=all_timestamps)



# Fill in the actual prices where we have them
call_prices.loc[call_df['timestamp'], 'price'] = call_df['close'].values
put_prices.loc[put_df['timestamp'], 'price'] = put_df['close'].values
stk_prices.loc[stk_df['timestamp'], 'price'] = stk_df['close'].values

# Forward fill missing values (use previous price when no new price available)
call_prices = call_prices.ffill()
put_prices = put_prices.ffill()
stk_prices = stk_prices.ffill()

# Backward fill missing values (use next price when no new price available)
call_prices = call_prices.bfill()
put_prices = put_prices.bfill()
stk_prices = stk_prices.bfill()


# Build underlying_prices Series
underlying_prices = pd.Series(stk_prices['price'], index=pd.to_datetime(stk_prices.index, unit='ms'))

# Build option_prices DataFrame with MultiIndex
option_data = []
for timestamp in all_timestamps:
    dt = pd.to_datetime(timestamp, unit='ms')
    # Add call option data if we have a price
    if not pd.isna(call_prices.loc[timestamp, 'price']):
        option_data.append({
            'timestamp': dt,
            'contract_id': c_option_symbol,
            'mid_price': call_prices.loc[timestamp, 'price'],
            'strike': call_data['strike'],
            'expiry': call_data['expiration_date'],
            'type': call_data['option_type']
        })
    # Add put option data if we have a price
    if not pd.isna(put_prices.loc[timestamp, 'price']):
        option_data.append({
            'timestamp': dt,
            'contract_id': p_option_symbol,
            'mid_price': put_prices.loc[timestamp, 'price'],
            'strike': put_data['strike'],
            'expiry': put_data['expiration_date'],
            'type': put_data['option_type']
        })

# Convert to DataFrame and set MultiIndex
options_df = pd.DataFrame(option_data)
option_prices = options_df.set_index(['timestamp', 'contract_id'])

# ---- INSTANTIATE THE OptionHedger ----
hedger = OptionHedger(
    underlying_prices=underlying_prices,
    option_prices=option_prices,
    r=r,
    q=0.0,
    delta_tolerance=0.01
)
print(underlying_prices.head())
print(option_prices.head())

# ---- OPEN AN INITIAL STRADDLE AT t0 (1 CALL + 1 PUT) ----
t0 = pd.to_datetime(call_prices.index[0], unit='ms')
hedger.open_position(timestamp=t0, contract_id=c_option_symbol, quantity=1)  # long 1 CALL
hedger.open_position(timestamp=t0, contract_id=p_option_symbol, quantity=1)  # long 1 PUT

print(f"Opened straddle at {t0}:")
print(f"  Call: {c_option_symbol}, Strike: {call_data['strike']}, Expiry: {call_data['expiration_date'].date()}")
print(f"  Put: {p_option_symbol}, Strike: {put_data['strike']}, Expiry: {put_data['expiration_date'].date()}")
print(f"  Underlying Price: ${underlying_prices.loc[t0]:.2f}")
print("  Starting hedged shares:", hedger.hedged_shares)
print("  Starting cash balance: $%.2f" % hedger.cash)


# ---- STEP THROUGH THE NEXT 10 MINUTES ----
print("\nStepping forward 10 time steps:\n")
for step in range(10):
    hedger.forward()
    row = hedger.timeline[-1]
    print(f"Step {step+1} at {row['timestamp']}:")
    print(f"  S_t = ${row['S_t']:.2f}")
    print(f"  Net Δ (pre-hedge) = {row['net_delta_pre_hedge']:.4f}")
    print(f"  Δ Traded = {row['delta_traded']:.4f}")
    print(f"  Hedge Shares = {row['hedged_shares']:.4f}")
    print(f"  MTM PnL = ${row['mtm_pnl']:.2f}\n")
