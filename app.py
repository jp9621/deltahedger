# app.py

import time
import calendar
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
from polygon import RESTClient
from OptionHedger import OptionHedger

# --------------------------------
# 1) Helper: parse "Month Year" → date range in milliseconds
# --------------------------------
def month_to_date_range(user_str: str):
    try:
        dt = datetime.strptime(user_str, "%B %Y")
        year, month = dt.year, dt.month
    except ValueError:
        dt = datetime.strptime(user_str, "%Y-%m")
        year, month = dt.year, dt.month

    last_day = calendar.monthrange(year, month)[1]
    start_date = datetime(year, month, 1, 0, 0, 0)
    end_date   = datetime(year, month, last_day, 23, 59, 59)
    
    # Convert to millisecond timestamps
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    return start_ms, end_ms

# --------------------------------
# 2) Helper: fetch 4H bars for a ticker
# --------------------------------
def fetch_bars(ticker: str, start_ms: int, end_ms: int, api_key: str):
    """Fetch 1-hour bars for a ticker and return DataFrame with millisecond timestamp index."""
    client = RESTClient(api_key=api_key)
    
    # Convert ms to date strings for Polygon API
    from_date = datetime.fromtimestamp(start_ms / 1000).strftime("%Y-%m-%d")
    to_date = datetime.fromtimestamp(end_ms / 1000).strftime("%Y-%m-%d")

    agg_resp = client.get_aggs(
        ticker=ticker,
        multiplier=1,            # 1 hour
        timespan="hour",
        from_=from_date,
        to=to_date,
        adjusted=True,
        sort="asc",
        limit=50000
    )

    if not agg_resp:
        raise ValueError(f"No hourly data for {ticker} between {from_date} and {to_date}.")
    
    df = pd.DataFrame(agg_resp)
    # Keep timestamps as milliseconds
    df.index = df["timestamp"]
    return df

# --------------------------------
# 3) Helper: pick ATM straddle
# --------------------------------
def pick_atm_straddle(ticker: str, 
                      first_bar_price: float, 
                      expiry_year: int, 
                      expiry_month: int, 
                      api_key: str):
    """
    Returns contract symbols and expiry in milliseconds since epoch.
    """
    atm_strike = round(first_bar_price)

    if expiry_month == 12:
        next_month = 1
        next_year = expiry_year + 1
    else:
        next_month = expiry_month + 1
        next_year = expiry_year

    # Find the 3rd Friday of the next month
    first_day = datetime(next_year, next_month, 1)
    fridays = [d for d in range(1, 22) if datetime(next_year, next_month, d).weekday() == 4]
    third_friday = fridays[2]
    expiry_dt = datetime(next_year, next_month, third_friday)
    expiry_ms = int(expiry_dt.timestamp() * 1000)

    # Format YYMMDD for contract symbol
    expiry_str = f"{next_year%100:02d}{next_month:02d}{third_friday:02d}"
    c_symbol = f"O:{ticker}{expiry_str}C{atm_strike*1000:08d}"
    p_symbol = f"O:{ticker}{expiry_str}P{atm_strike*1000:08d}"

    return c_symbol, p_symbol, atm_strike, expiry_ms

# --------------------------------
# 4) Streamlit UI
# --------------------------------
st.title("Delta-Hedger Demo")

# 4.1) API key
API_KEY = 'w2chIPf4EUplQqQv6b8Nxnn8GQV8pfGC'

# 4.2) Inputs: Month & Ticker
month_str = st.text_input("Historical Month (e.g. January 2024 or 2024-01)")
ticker    = st.text_input("Underlying Ticker (e.g. AAPL)").upper()

if not month_str or not ticker:
    st.info("Enter both a month and a ticker, then click Run Demo.")
    st.stop()

# 4.3) Run Button
if st.button("Run Demo"):
    try:
        # 5.1) Parse month → date range in milliseconds
        start_ms, end_ms = month_to_date_range(month_str)
        start_date = datetime.fromtimestamp(start_ms / 1000).date()
        end_date = datetime.fromtimestamp(end_ms / 1000).date()
        st.write(f"Running demo for **{ticker}** from {start_date} to {end_date}")

        # 5.2) Fetch hourly bars for underlying
        underlying_df = fetch_bars(ticker, start_ms, end_ms, API_KEY)
        
        # 5.3) Choose ATM straddle based on first hour's price
        first_price = underlying_df.iloc[0]["close"]
        expiry_year = datetime.fromtimestamp(start_ms / 1000).year
        expiry_month = datetime.fromtimestamp(start_ms / 1000).month
        c_sym, p_sym, atm_strike, expiry_ms = pick_atm_straddle(
            ticker, first_price, expiry_year, expiry_month, API_KEY
        )
        expiry_date = datetime.fromtimestamp(expiry_ms / 1000).date()
        st.write(f"ATM Strike chosen: {atm_strike}, Call: {c_sym}, Put: {p_sym}, Expiry ≈ {expiry_date}")

        # 5.4) Fetch hourly bars for call & put
        call_df = fetch_bars(c_sym, start_ms, end_ms, API_KEY)
        put_df = fetch_bars(p_sym, start_ms, end_ms, API_KEY)

        # 5.5) Get all unique timestamps across all assets
        all_timestamps = pd.Series(
            sorted(set(underlying_df.index.tolist() + 
                      call_df.index.tolist() + 
                      put_df.index.tolist()))
        )

        # Create DataFrames indexed by timestamp for each asset
        call_prices = pd.DataFrame(index=all_timestamps)
        put_prices = pd.DataFrame(index=all_timestamps)
        underlying_prices = pd.DataFrame(index=all_timestamps)

        # Fill in the actual prices where we have them
        call_prices.loc[call_df.index, 'price'] = call_df['close'].values
        put_prices.loc[put_df.index, 'price'] = put_df['close'].values
        underlying_prices.loc[underlying_df.index, 'price'] = underlying_df['close'].values

        # Forward fill missing values
        call_prices = call_prices.ffill().bfill()
        put_prices = put_prices.ffill().bfill()
        underlying_prices = underlying_prices.ffill()

        # Find first valid stock price timestamp
        first_valid_ts = underlying_prices.first_valid_index()
        valid_timestamps = all_timestamps[all_timestamps >= first_valid_ts]

        # Build option_prices DataFrame with MultiIndex
        option_data = []
        for ts in valid_timestamps:
            # Add call option data
            option_data.append({
                'timestamp': ts,
                'contract_id': c_sym,
                'mid_price': call_prices.loc[ts, 'price'],
                'strike': atm_strike,
                'expiry': expiry_ms,  # Now in milliseconds
                'type': 'call'
            })
            # Add put option data
            option_data.append({
                'timestamp': ts,
                'contract_id': p_sym,
                'mid_price': put_prices.loc[ts, 'price'],
                'strike': atm_strike,
                'expiry': expiry_ms,  # Now in milliseconds
                'type': 'put'
            })

        # Convert to DataFrame and set MultiIndex
        options_df = pd.DataFrame(option_data)
        options_df = options_df.set_index(['timestamp', 'contract_id'])

        # Create underlying Series starting from first valid timestamp
        underlying_series = pd.Series(
            underlying_prices.loc[valid_timestamps, 'price'],
            index=valid_timestamps,
            name='price'
        )

        # 5.6) Instantiate OptionHedger
        hedger = OptionHedger(
            underlying_prices=underlying_series,
            option_prices=options_df,
            r=0.01,
            q=0.0,
            delta_tolerance=0.01
        )

        # 5.7) Set initial timestamp
        t0 = valid_timestamps[0]
        t0_idx = list(hedger.underlying_prices.index).index(t0)
        hedger.current_step = t0_idx

        # 5.8) Open ATM straddle at t0
        hedger.open_position(t0, c_sym, 1)
        hedger.open_position(t0, p_sym, 1)

        st.success(f"Opened ATM straddle at {datetime.fromtimestamp(t0/1000)} → Sₜ₀=${underlying_series[t0]:.2f}, Strike={atm_strike}")

        # 6) Prepare containers for real-time animation
        price_chart = st.empty()
        delta_chart = st.empty()
        iv_chart    = st.empty()
        console     = st.empty()

        # Lists to accumulate values as we step
        ts_list         = []
        price_list      = []
        delta_pre_list  = []
        delta_post_list = []  # New list for post-hedge delta
        delta_traded_list = []
        hedge_list      = []
        mtm_list        = []
        iv_list         = []

        # 7) Simulation loop: Step through each 4H bar
        total_steps = len(hedger.underlying_prices) - t0_idx
        st.info(f"Stepping through {total_steps} 4-hour bars...")

        for step in range(total_steps - 1):
            hedger.forward()
            row = hedger.timeline[-1]
            t_ms = row["timestamp"]  # Now in milliseconds
            S_t = row["S_t"]
            net_pre = row["net_delta_pre_hedge"]
            net_post = row["net_delta_post_hedge"]  # Get post-hedge delta
            delta_traded = row["delta_traded"]
            hedged_shares = row["hedged_shares"]
            mtm = row["mtm_pnl"]

            # If you want the implied vol at that timestamp, compute it via the private method:
            opt_slice = options_df.loc[t_ms]
            iv_dict = hedger._compute_implied_vols(opt_slice, S_t)
            iv_avg = None
            if c_sym in iv_dict and p_sym in iv_dict:
                iv_avg = 0.5 * (iv_dict[c_sym] + iv_dict[p_sym])
            elif c_sym in iv_dict:
                iv_avg = iv_dict[c_sym]
            elif p_sym in iv_dict:
                iv_avg = iv_dict[p_sym]
            else:
                iv_avg = float("nan")

            # 8) Append to our lists
            ts_list.append(t_ms)
            price_list.append(S_t)
            delta_pre_list.append(net_pre)
            delta_post_list.append(net_post)  # Store post-hedge delta
            delta_traded_list.append(delta_traded)
            hedge_list.append(hedged_shares)
            mtm_list.append(mtm)
            iv_list.append(iv_avg)

            # 9) Update the three charts and console in real time
            #  9a) Price chart
            with price_chart:
                st.subheader("Underlying Price (4H Bars)")
                df_price = pd.DataFrame({
                    "timestamp": [datetime.fromtimestamp(ts/1000) for ts in ts_list],
                    "price": price_list
                }).set_index("timestamp")
                fig_price = px.line(df_price, labels={"timestamp":"Time", "price":"Price"}, 
                                    title="Underlying Price")
                st.plotly_chart(fig_price, use_container_width=True)

            #  9b) Delta chart (pre vs post)
            with delta_chart:
                st.subheader("Δ pre-hedge vs. Δ post-hedge")
                df_delta = pd.DataFrame({
                    "timestamp": [datetime.fromtimestamp(ts/1000) for ts in ts_list],
                    "delta_pre": delta_pre_list,
                    "delta_post": delta_post_list  # Use actual post-hedge delta
                }).set_index("timestamp")
                fig_delta = px.line(df_delta, 
                                    labels={"value":"Delta", "variable":"Legend"},
                                    title="Net Δ Pre-Hedge vs. Post-Hedge")
                st.plotly_chart(fig_delta, use_container_width=True)

            #  9c) IV chart
            with iv_chart:
                st.subheader("Implied Volatility (avg of call & put)")
                df_iv = pd.DataFrame({
                    "timestamp": [datetime.fromtimestamp(ts/1000) for ts in ts_list],
                    "iv": iv_list
                }).set_index("timestamp")
                fig_iv = px.line(df_iv, labels={"iv":"Implied Vol"}, title="Implied Vol over Time")
                st.plotly_chart(fig_iv, use_container_width=True)

            #  9d) Console
            t_str = datetime.fromtimestamp(t_ms/1000).strftime("%Y-%m-%d %H:%M")
            line = (
                f"Step {step+1} @ {t_str}  |  "
                f"Sₜ=${S_t:.2f}  |  "
                f"Δ_pre={net_pre:.4f}  |  "
                f"Δ_traded={delta_traded:.4f}  |  "
                f"HedgeShares={hedged_shares:.4f}  |  "
                f"MTM=${mtm:.2f}\n"
            )
            if step == 0:
                log_text = line
            else:
                log_text += line
            console.text(log_text)

            time.sleep(0.01)

        st.success("✅ Backtest complete!")

    except Exception as e:
        st.error(f"Error during backtest: {e}")
        st.stop()
