# OptionHedger Implementation

A Python implementation of a delta-neutral options hedging system using the Black-Scholes model.

`https://deltahedgerdemo.streamlit.app/`

## Overview

The `OptionHedger` class implements a backtester-style manager that:
- Maintains historical price data for options and their underlying assets
- Tracks open positions (both option legs and underlying hedge positions)
- Computes implied volatilities and option Greeks
- Simulates delta-neutral hedging by rebalancing underlying positions

## Core Components

### Position Management
- Tracks multiple option positions ("legs")
- Maintains underlying hedge positions
- Keeps track of cash balance and mark-to-market P&L

### Greeks Calculation
- Implements Black-Scholes model for option pricing
- Calculates option deltas for both calls and puts
- Handles special cases like at-expiry options

### Delta Hedging
- Computes net portfolio delta
- Automatically rebalances when delta exceeds tolerance
- Tracks hedging performance and trade history

## Key Methods

### `__init__`
```python
def __init__(self, underlying_prices, option_prices, r=0.005, q=0.0, delta_tolerance=0.01)
```
- `underlying_prices`: Series of underlying asset prices
- `option_prices`: DataFrame with option data (price, strike, expiry, type)
- `r`: Risk-free rate
- `q`: Dividend yield
- `delta_tolerance`: Maximum allowed delta exposure

### `forward()`
Steps forward in time and:
1. Computes current portfolio delta
2. Rebalances if needed
3. Updates mark-to-market values
4. Records metrics in timeline

### `open_position()`
```python
def open_position(self, timestamp_ms, contract_id, quantity)
```
Opens a new option position with specified:
- Timestamp
- Contract ID
- Quantity (+ve for long, -ve for short)

### Black-Scholes Implementation
- `_bs_price()`: Computes theoretical option price
- `_bs_delta()`: Calculates option delta
- `_compute_implied_vols()`: Derives implied volatility using bisection method

## Data Structures

### Option Leg Format
```python
{
    'contract_id': str,      # Option identifier
    'type': str,            # 'call' or 'put'
    'strike': float,        # Strike price
    'expiry': int,          # Expiry in ms timestamp
    'entry_price': float,   # Initial price
    'quantity': int         # Position size
}
```

### Timeline Entry Format
```python
{
    'timestamp': int,       # Current time in ms
    'S_t': float,          # Underlying price
    'net_delta_pre_hedge': float,  # Delta before hedging
    'net_delta_post_hedge': float, # Delta after hedging
    'delta_traded': float,  # Hedge adjustment
    'hedged_shares': float, # Total hedge position
    'mtm_pnl': float       # Mark-to-market P&L
}
```

## Notes

- All timestamps are stored as integer milliseconds since epoch
- Special handling for expired options using intrinsic value
- Implied volatility calculation uses numerical methods (bisection)
- Delta hedging assumes perfect liquidity and no transaction costs
