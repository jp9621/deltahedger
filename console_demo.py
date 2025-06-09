import datetime
import math
import re

import pandas as pd
from polygon import RESTClient
from scipy.stats import norm

from OptionHedger import OptionHedger


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


API_KEY = 'w2chIPf4EUplQqQv6b8Nxnn8GQV8pfGC'
c_option_symbol = 'O:SPY250630C00600000'
p_option_symbol = 'O:SPY250630P00595000'
date_str = '2025-05-23'
r = 0.01


call_data = OptionHedger.parse_option_symbol(c_option_symbol)
put_data = OptionHedger.parse_option_symbol(p_option_symbol)

underlying_sym = call_data['underlying_sym']

start_ms, end_ms = OptionHedger.get_day_bounds_unix_ms(date_str)
expiration_ms = int(call_data['expiration_date'].timestamp() * 1000)


client = RESTClient(api_key=API_KEY)


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


call_df = pd.DataFrame(call_resp)
put_df = pd.DataFrame(put_resp)
stk_df = pd.DataFrame(stk_resp)

all_timestamps = pd.Series(
    sorted(set(call_df['timestamp'].tolist() + 
             put_df['timestamp'].tolist() + 
             stk_df['timestamp'].tolist()))
)

call_prices = pd.DataFrame(index=all_timestamps)
put_prices = pd.DataFrame(index=all_timestamps)
stk_prices = pd.DataFrame(index=all_timestamps)



call_prices.loc[call_df['timestamp'], 'price'] = call_df['close'].values
put_prices.loc[put_df['timestamp'], 'price'] = put_df['close'].values
stk_prices.loc[stk_df['timestamp'], 'price'] = stk_df['close'].values

call_prices = call_prices.ffill().bfill()
put_prices = put_prices.ffill().bfill()
stk_prices = stk_prices.ffill()

first_valid_stock_ts = stk_prices.first_valid_index()
print("Stock price at first_valid_stock_ts:", stk_prices.loc[first_valid_stock_ts, 'price'])
valid_timestamps = all_timestamps[all_timestamps >= first_valid_stock_ts]
underlying_prices = pd.Series(stk_prices.loc[valid_timestamps, 'price'], 
                            index=valid_timestamps)

option_data = []
for timestamp in valid_timestamps:
    if not pd.isna(call_prices.loc[timestamp, 'price']):
        option_data.append({
            'timestamp': timestamp,
            'contract_id': c_option_symbol,
            'mid_price': call_prices.loc[timestamp, 'price'],
            'strike': call_data['strike'],
            'expiry': call_data['expiration_date'],
            'type': call_data['option_type']
        })
    if not pd.isna(put_prices.loc[timestamp, 'price']):
        option_data.append({
            'timestamp': timestamp,
            'contract_id': p_option_symbol,
            'mid_price': put_prices.loc[timestamp, 'price'],
            'strike': put_data['strike'],
            'expiry': put_data['expiration_date'],
            'type': put_data['option_type']
        })

options_df = pd.DataFrame(option_data)
option_prices = options_df.set_index(['timestamp', 'contract_id'])

hedger = OptionHedger(
    underlying_prices=underlying_prices,
    option_prices=option_prices,
    r=r,
    q=0.0,
    delta_tolerance=0.01
)

call_times = set(option_prices.xs(c_option_symbol, level='contract_id', drop_level=False).index.get_level_values(0))
put_times = set(option_prices.xs(p_option_symbol, level='contract_id', drop_level=False).index.get_level_values(0))
common_times = sorted(list(call_times & put_times))

if not common_times:
    raise ValueError("No common timestamps between call and put options!")

t0 = common_times[0]

t0_idx = list(hedger.underlying_prices.index).index(t0)
hedger.current_step = t0_idx

print(f"\nOpening straddle at timestamp {t0}:")
print(f"  Call: {c_option_symbol}, Strike: {call_data['strike']}, Expiry: {call_data['expiration_date'].date()}")
print(f"  Put: {p_option_symbol}, Strike: {put_data['strike']}, Expiry: {put_data['expiration_date'].date()}")
print(f"  Underlying Price: ${underlying_prices.loc[t0]:.2f}")

hedger.open_position(t0, c_option_symbol, 1)
hedger.open_position(t0, p_option_symbol, 1)

print("  Starting hedged shares:", hedger.hedged_shares)
print("  Starting cash balance: $%.2f" % hedger.cash)


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
