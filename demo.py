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


# ---- PARSE OPTION SYMBOL ----
parsed_data = OptionHedger.parse_option_symbol(c_option_symbol)
underlying_sym = parsed_data['underlying_sym']
expiration_date = parsed_data['expiration_date']
option_type = parsed_data['option_type']
K = parsed_data['strike']


start_ms, end_ms = OptionHedger.get_day_bounds_unix_ms(date_str)
expiration_ms = int(expiration_date.timestamp() * 1000)


# ---- INSTANTIATE POLYGON CLIENT ----
client = RESTClient(api_key=API_KEY)


# ---- FETCH 1-MINUTE AGGREGATES FOR OPTION & UNDERLYING ----
opt_resp = client.get_aggs(
    ticker=c_option_symbol,
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


# ---- CONVERT TO DATAFRAMES ----
opt_df = pd.DataFrame(opt_resp)   # raw option minute-bars
stk_df = pd.DataFrame(stk_resp)   # raw underlying minute-bars


# ---- TURN MILLISECONDS → DATETIME INDEX ----
opt_df['dt'] = pd.to_datetime(opt_df['timestamp'], unit='ms')
stk_df['dt'] = pd.to_datetime(stk_df['timestamp'], unit='ms')


# ---- MERGE ON EXACT MINUTE STAMPS ----
#    Keep only rows where the option bar and underlying bar share the same 'dt'
merged = pd.merge(
    opt_df,
    stk_df[['dt','close']].rename(columns={'close':'stk_close'}),
    on='dt', 
    how='inner'
)


# ---- BUILD underlying_prices: pd.Series indexed by dt ----
underlying_prices = merged.set_index('dt')['stk_close'].copy()


# ---- BUILD option_prices: MultiIndex DataFrame ----
# Index = (dt, option_symbol), Columns = ['mid_price', 'strike', 'expiry', 'type']
option_index = pd.MultiIndex.from_tuples(
    [(row.dt, c_option_symbol) for row in merged.itertuples()],
    names=['timestamp','contract_id']
)
option_prices = pd.DataFrame(index=option_index, columns=['mid_price','strike','expiry','type'])
option_prices['mid_price'] = merged['close'].values
option_prices['strike']    = K
option_prices['expiry']    = expiration_date
option_prices['type']      = option_type


# ---- INSTANTIATE THE OptionHedger ----
hedger = OptionHedger(
    underlying_prices=underlying_prices,
    option_prices=option_prices,
    r=r,
    q=0.0,
    delta_tolerance=0.01
)


# ---- OPEN AN INITIAL STRADDLE AT t0 (1 CALL + 1 PUT) ----
t0 = hedger.current_timestamp
hedger.open_position(timestamp=t0, contract_id=c_option_symbol, quantity=1)  # long 1 CALL
hedger.open_position(timestamp=t0, contract_id=p_option_symbol, quantity=1)  # long 1 PUT

print(f"Opened straddle at {t0}:")
print(f"  Contract: {c_option_symbol}, Strike: {K}, Expiry: {expiration_date.date()}")
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
