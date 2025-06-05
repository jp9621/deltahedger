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
# 1) Helper: parse “Month Year” → date range
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
    return start_date, end_date

# --------------------------------
# 2) Helper: fetch 4H bars for a ticker
# --------------------------------
def fetch_4h_bars(ticker: str, start: datetime, end: datetime, api_key: str):
    client = RESTClient(api_key=api_key)
    from_date = start.strftime("%Y-%m-%dT%H:%M:%S")
    to_date   = end.strftime("%Y-%m-%dT%H:%M:%S")

    agg_resp = client.get_aggs(
        ticker=ticker,
        multiplier=1,            # 1 hour
        timespan="hour",
        _from=from_date,
        to=to_date,
        adjusted=True,
        sort="asc",
        limit=50000
    )

    df = pd.DataFrame(agg_resp)
    if df.empty:
        raise ValueError(f"No hourly data for {ticker} between {start} and {end}.")

    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    bars_4h = df["close"].resample("4H").last().ffill()
    return bars_4h

# --------------------------------
# 3) Helper: pick ATM straddle (stub)
# --------------------------------
def pick_atm_straddle(ticker: str, 
                      first_bar_price: float, 
                      expiry_year: int, 
                      expiry_month: int, 
                      api_key: str):
    """
    This is a stub. 
    In practice, you would:
      1) query client.get_option_contracts(…)
      2) Filter to the expiration (expiry_year, expiry_month).
      3) Find the strike closest to 'first_bar_price'.
      4) Return the full Polygon symbols for call & put.
    For now, assume you already know that logic and just return placeholders.
    """
    # ── Example fallback: round to nearest dollar for ATM:
    atm_strike = round(first_bar_price)

    # The user can fill this in with actual contract lookup:
    c_symbol = f"O:{ticker}{expiry_year%100:02d}{expiry_month:02d}01C{atm_strike*1000:08d}"
    p_symbol = f"O:{ticker}{expiry_year%100:02d}{expiry_month:02d}01P{atm_strike*1000:08d}"
    expiry_dt = datetime(expiry_year, expiry_month, 1)  # approximate; replace with actual expiry date

    return c_symbol, p_symbol, atm_strike, expiry_dt

# --------------------------------
# 4) Streamlit UI
# --------------------------------
st.title("Δ-Hedger Backtester (4-Hour Bars)")

# 4.1) API key
API_KEY = st.text_input("Polygon API Key", type="password")
if not API_KEY:
    st.warning("Please enter your Polygon API Key to proceed.")
    st.stop()

# 4.2) Inputs: Month & Ticker
month_str = st.text_input("Historical Month (e.g. January 2024 or 2024-01)")
ticker    = st.text_input("Underlying Ticker (e.g. AAPL)").upper()

if not month_str or not ticker:
    st.info("Enter both a month and a ticker, then click Run Backtest.")
    st.stop()

# 4.3) Run Button
if st.button("Run Backtest"):
    try:
        # 5.1) Parse month → date range
        start_date, end_date = month_to_date_range(month_str)
        st.write(f"Running backtest for **{ticker}** from {start_date.date()} to {end_date.date()}")

        # 5.2) Fetch 4H bars for underlying
        underlying_4h = fetch_4h_bars(ticker, start_date, end_date, API_KEY)

        # 5.3) Choose ATM straddle at the very first 4H bar
        first_ts = underlying_4h.index[0]
        first_price = underlying_4h.iloc[0]
        # We choose the _next_ monthly expiry after start_date:
        # e.g. if user asked "Jan 2024", expiry_year=2024, expiry_month=1.
        # You could instead look up the actual 3rd Friday, etc. Here we stub:
        expiry_year  = start_date.year
        expiry_month = start_date.month
        c_sym, p_sym, atm_strike, expiry_dt = pick_atm_straddle(
            ticker, first_price, expiry_year, expiry_month, API_KEY
        )
        st.write(f"ATM Strike chosen: {atm_strike}, Call: {c_sym}, Put: {p_sym}, Expiry ≈ {expiry_dt.date()}")

        # 5.4) Fetch 4H bars for call & put
        call_4h = fetch_4h_bars(c_sym, start_date, end_date, API_KEY)
        put_4h  = fetch_4h_bars(p_sym, start_date, end_date, API_KEY)

        # 5.5) Build option_prices DataFrame
        opt_rows = []
        for ts, px in call_4h.items():
            opt_rows.append({
                "timestamp": ts,
                "contract_id": c_sym,
                "mid_price": px,
                "strike": atm_strike,
                "expiry": expiry_dt,
                "type": "call"
            })
        for ts, px in put_4h.items():
            opt_rows.append({
                "timestamp": ts,
                "contract_id": p_sym,
                "mid_price": px,
                "strike": atm_strike,
                "expiry": expiry_dt,
                "type": "put"
            })
        options_df = pd.DataFrame(opt_rows).set_index(["timestamp", "contract_id"]).sort_index()

        # 5.6) Instantiate OptionHedger
        hedger = OptionHedger(
            underlying_prices=underlying_4h.rename("price"),
            option_prices=options_df,
            r=0.01,
            q=0.0,
            delta_tolerance=0.01
        )

        # 5.7) Find index of first_ts in hedger.timestamps, set t0
        t0_idx = list(hedger.underlying_prices.index).index(first_ts)
        hedger.current_step = t0_idx

        # 5.8) Open ATM straddle at t0
        hedger.open_position(first_ts, c_sym, 1)
        hedger.open_position(first_ts, p_sym, 1)

        st.success(f"Opened ATM straddle at {first_ts} → Sₜ₀=${first_price:.2f}, Strike={atm_strike}")

        # 6) Prepare containers for real-time animation
        price_chart = st.empty()
        delta_chart = st.empty()
        iv_chart    = st.empty()
        console     = st.empty()

        # Lists to accumulate values as we step
        ts_list         = []
        price_list      = []
        delta_pre_list  = []
        delta_traded_list = []
        hedge_list      = []
        mtm_list        = []
        iv_list         = []

        # 7) Simulation loop: Step through each 4H bar
        total_steps = len(hedger.underlying_prices) - t0_idx
        st.info(f"Stepping through {total_steps} 4-hour bars...")

        for step in range(total_steps):
            hedger.forward()
            row = hedger.timeline[-1]
            t   = row["timestamp"]
            S_t = row["S_t"]
            net_pre = row["net_delta_pre_hedge"]
            delta_traded = row["delta_traded"]
            hedged_shares = row["hedged_shares"]
            mtm = row["mtm_pnl"]

            # If you want the implied vol at that timestamp, compute it via the private method:
            # (NOTE: OptionHedger doesn’t currently return IVs, so we re‐compute it here.)
            # 1) Extract the option_slice at time t:
            opt_slice = options_df.loc[t]  # DataFrame of shape (2 rows)
            iv_dict = hedger._compute_implied_vols(opt_slice, S_t)
            # average of the two IVs (call & put):
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
            ts_list.append(t)
            price_list.append(S_t)
            delta_pre_list.append(net_pre)
            delta_traded_list.append(delta_traded)
            hedge_list.append(hedged_shares)
            mtm_list.append(mtm)
            iv_list.append(iv_avg)

            # 9) Update the three charts and console in real time
            #  9a) Price chart
            with price_chart:
                st.subheader("Underlying Price (4H Bars)")
                df_price = pd.DataFrame({
                    "timestamp": ts_list,
                    "price": price_list
                }).set_index("timestamp")
                fig_price = px.line(df_price, labels={"timestamp":"Time", "price":"Price"}, 
                                    title="Underlying Price")
                st.plotly_chart(fig_price, use_container_width=True)

            #  9b) Delta chart (pre vs post)
            with delta_chart:
                st.subheader("Δ pre-hedge vs. Δ post-hedge")
                df_delta = pd.DataFrame({
                    "timestamp": ts_list,
                    "delta_pre": delta_pre_list,
                    "delta_post": [0.0]*len(ts_list)
                }).set_index("timestamp")
                fig_delta = px.line(df_delta, 
                                    labels={"value":"Delta", "variable":"Legend"},
                                    title="Net Δ Pre-Hedge (blue) vs. Post-Hedge = 0 (red)")
                st.plotly_chart(fig_delta, use_container_width=True)

            #  9c) IV chart
            with iv_chart:
                st.subheader("Implied Volatility (avg of call & put)")
                df_iv = pd.DataFrame({
                    "timestamp": ts_list,
                    "iv": iv_list
                }).set_index("timestamp")
                fig_iv = px.line(df_iv, labels={"iv":"Implied Vol"}, title="Implied Vol over Time")
                st.plotly_chart(fig_iv, use_container_width=True)

            #  9d) Console
            line = (
                f"Step {step+1} @ {t}  |  "
                f"Sₜ=${S_t:.2f}  |  "
                f"Δ_pre={net_pre:.4f}  |  "
                f"Δ_traded={delta_traded:.4f}  |  "
                f"HedgeShares={hedged_shares:.4f}  |  "
                f"MTM=${mtm:.2f}\n"
            )
            # Accumulate lines
            if step == 0:
                log_text = line
            else:
                log_text += line
            console.text(log_text)

            #  9e) Sleep so the user sees it “animate”
            time.sleep(0.5)

        st.success("✅ Backtest complete!")

    except Exception as e:
        st.error(f"Error during backtest: {e}")
        st.stop()
