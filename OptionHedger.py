import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
import datetime
import re


class OptionHedger:
    """
    A backtester‐style manager that:
      - Holds a full option chain's historical prices (for every timestamp).
      - Keeps track of open positions (option legs + underlying hedge).
      - On each call to forward(), updates net delta and rebalances underlying if needed.
    """

    def __init__(
        self,
        underlying_prices: pd.Series,
        option_prices: pd.DataFrame,
        r: float = 0.005,
        q: float = 0.0,
        delta_tolerance: float = 0.01
    ):
        """
        Parameters
        ----------
        underlying_prices : pd.Series
            Indexed by timestamp, e.g. DatetimeIndex, containing the underlying mid price S_t.

        option_prices : pd.DataFrame
            A MultiIndex DataFrame indexed by (timestamp, contract_id), where each row has:
              - 'mid_price'  (e.g. average of bid/ask or mid), 
              - 'strike', 
              - 'expiry' (as a pd.Timestamp), 
              - 'type'   ('call' or 'put').
            In other words:
                index = pd.MultiIndex.from_product([timestamps, contract_ids])
                columns = ['mid_price', 'strike', 'expiry', 'type']

            This gives you the full chain at each timestamp for every contract you care about.
            (You could also store these attributes separately, but a tidy MultiIndex is convenient.)

        r : float
            Risk‐free interest rate (annual, e.g. 0.005 = 0.5%).

        q : float
            Dividend yield of the underlying (annual).

        delta_tolerance : float
            If |net_delta| > delta_tolerance, then the forward() call will trigger a hedge trade.
        """
        # 1) Store market data
        self.underlying_prices = underlying_prices.copy()
        self.option_prices     = option_prices.copy()

        # 2) Sort indexes just to be sure
        self.underlying_prices = self.underlying_prices.sort_index()
        self.option_prices     = self.option_prices.sort_index()

        # 3) Risk parameters
        self.r = r
        self.q = q

        # 4) Hedging tolerance
        self.delta_tolerance = delta_tolerance

        # 5) Internal state
        #   - Keep track of which timestamp we're at (integer pointer into the sorted index)
        self.timestamps = list(self.underlying_prices.index)
        self.current_step = 0

        # 6) Positions
        #   - A list of dicts: each dict = { 'contract_id', 'type', 'strike', 'expiry', 
        #                                     'entry_price', 'quantity' }
        #   - quantity > 0 means long; < 0 means short.
        self.open_legs = []

        # 7) Hedge: how many shares of underlying we currently hold
        self.hedged_shares = 0.0

        # 8) Cash ledger (option entry/exit, plus underlying trades)
        self.cash = 0.0

        # 9) Logs: keep per‐step records if you want to export later
        self.timeline = []  # will append a dict at each forward() call
        self.hedge_log = [] # records of actual hedge trades {timestamp, delta_before, delta_traded, price}


    def forward(self):
        """
        Advance to the next timestamp. This method does:
          1) Load latest S_t and all option mid_prices for that timestamp.
          2) Compute implied vol for each alive contract (using mid_price).
          3) Compute Greeks (especially delta) for each open leg using current S_t, TTM, implied vol.
          4) Sum net delta (∑ delta_i * qty_i) - hedged_shares.
          5) If |net_delta| > delta_tolerance, trade underlying to bring net_delta → 0 (or within tolerance).
          6) Record PnL (unrealized on options + PnL on underlying hedge + cash).
          7) Increment current_step, append to logs.
        """
        # 0) If we're already at the final timestamp, do nothing
        if self.current_step >= len(self.timestamps):
            raise IndexError("Already at end of price series.")

        # 1) Get current timestamp, S_t, and entire option‐chain prices
        t = self.timestamps[self.current_step]
        S_t = self.underlying_prices.loc[t]

        # Extract all contracts' mid‐prices at this timestamp
        #   option_slice is a DataFrame indexed by contract_id with columns ['mid_price', 'strike', 'expiry', 'type']
        option_slice = self.option_prices.loc[t]  # this is a DataFrame (index=contract_id)

        # 2) For every open leg, compute its current delta using implied vol
        #    First, build a dict: contract_id → implied_vol
        iv_dict = self._compute_implied_vols(option_slice, S_t)

        # 3) Compute net delta of the portfolio
        net_delta = 0.0
        for leg in self.open_legs:
            cid    = leg['contract_id']
            qty    = leg['quantity']
            strike = leg['strike']
            expiry = leg['expiry']
            otype  = leg['type']

            # Time‐to‐maturity in years, assuming 252 trading days:
            T = max((expiry - t).days, 0) / 252

            if T <= 0:
                # Contract has expired → delta is intrinsic (0 or ±1)
                delta_i = self._intrinsic_delta(S_t, strike, otype)
            else:
                sigma = iv_dict.get(cid, None)
                # If we're don't have an IV (e.g. missing price, stale), you could skip or carry forward.
                if sigma is None:
                    delta_i = 0.0
                else:
                    delta_i = self._bs_delta(S_t, strike, T, sigma, otype)

            net_delta += qty * delta_i

        # Subtract the current hedge position
        net_delta -= self.hedged_shares

        # 4) If |net_delta| > tolerance, execute hedge
        delta_to_trade = 0.0
        if abs(net_delta) > self.delta_tolerance:
            delta_to_trade = -net_delta
            self._execute_hedge_trade(t, S_t, delta_to_trade)
            # After this, new net_delta = net_delta + delta_to_trade ≈ 0

        # 5) Mark‐to‐market all open legs + underlying position + cash
        mtm_pnl = self._mark_to_market(t, S_t, iv_dict)

        # 6) Record into timeline
        self.timeline.append({
            'timestamp': t,
            'S_t': S_t,
            'net_delta_pre_hedge': net_delta,
            'delta_traded': delta_to_trade,
            'hedged_shares': self.hedged_shares,
            'mtm_pnl': mtm_pnl
        })

        # 7) Advance pointer
        self.current_step += 1


    def open_position(self, timestamp: pd.Timestamp, contract_id: str, quantity: int):
        """
        Manually open a new option leg at a given timestamp.
        In a backtest, you would call this exactly once (or whenever your signal tells you).
        
        Parameters
        ----------
        timestamp : pd.Timestamp
            Must equal the current timestamp (i.e. timestamp == self.timestamps[self.current_step])
            so we know which market prices to charge.

        contract_id : str
            A key that exists under self.option_prices.columns. In a MultiIndex DataFrame,
            contract_id would be in the second level: (timestamp, contract_id).

        quantity : int
            Number of contracts: +1 for long, -1 for short.
        """
        t = self.timestamps[self.current_step]
        assert timestamp == t, "Can only open position at the current time index."

        # Pull from option_prices: strike, expiry, type, mid_price
        row = self.option_prices.loc[(t, contract_id)]
        strike, expiry, otype, mid_price = (
            row['strike'], row['expiry'], row['type'], row['mid_price']
        )

        # Charge cash for entry: quantity * mid_price * contract_multiplier (assume 1 for simplicity)
        self.cash -= quantity * mid_price

        # Append leg to open_legs
        self.open_legs.append({
            'contract_id': contract_id,
            'type': otype,
            'strike': strike,
            'expiry': expiry,
            'entry_price': mid_price,
            'quantity': quantity
        })


    def close_position(self, timestamp: pd.Timestamp, idx_in_open_legs: int):
        """
        Close one leg in self.open_legs by selling at the current mid_price.
        You must pass in the index of the leg in self.open_legs to remove.
        """
        t = self.timestamps[self.current_step]
        assert timestamp == t, "Can only close position at the current time index."

        leg = self.open_legs[idx_in_open_legs]
        cid    = leg['contract_id']
        qty    = leg['quantity']

        # Current mid_price
        mid_price = self.option_prices.loc[(t, cid), 'mid_price']

        # Cash from closing: receive qty * mid_price (if qty was positive, selling; if qty negative, buying back)
        self.cash += qty * mid_price

        # Remove leg
        self.open_legs.pop(idx_in_open_legs)


    # ─── INTERNAL / HELPER METHODS ────────────────────────────────────────────

    def _compute_implied_vols(self, option_slice: pd.DataFrame, S_t: float) -> dict:
        """
        Given the option_slice (all contracts at time t, with mid_price, strike, expiry, type),
        numerically invert Black‐Scholes to find implied volatility for each contract.

        Returns
        -------
        iv_dict : dict[contract_id → float]
            The discovered implied vol for each contract. If any contract fails to converge,
            drop it (i.e. do not include in iv_dict).
        """
        iv_dict = {}
        for cid, row in option_slice.iterrows():
            price  = row['mid_price']
            K      = row['strike']
            expiry = row['expiry']
            otype  = row['type']

            T = max((expiry - self.timestamps[self.current_step]).days, 0) / 252
            if T <= 0:
                continue

            # Define BS formula price(sigma) - market_price = 0
            def f(sigma):
                return self._bs_price(S_t, K, T, sigma, otype) - price

            try:
                # Use Brent's method on [1e-6, 5.0] as plausible vol range
                implied_vol = brentq(f, 1e-6, 5.0, maxiter=100, xtol=1e-6)
                iv_dict[cid] = implied_vol
            except (ValueError, RuntimeError):
                # Could not find a root → skip
                continue

        return iv_dict


    def _bs_price(self, S, K, T, sigma, otype):
        """
        Black‐Scholes price for European call/put.
        """
        r, q = self.r, self.q
        if T <= 0:
            return max(0.0, (S - K)) if otype == 'call' else max(0.0, (K - S))

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if otype == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


    def _bs_delta(self, S, K, T, sigma, otype):
        """
        ∂Price/∂S for European call/put.
        """
        r, q = self.r, self.q
        if T <= 0:
            # Expired: delta is 0 or ±1
            return 1.0 if (otype == 'call' and S > K) else 0.0 \
                   if otype == 'call' else -1.0 if (otype == 'put' and S < K) else 0.0

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if otype == 'call':
            return np.exp(-q * T) * norm.cdf(d1)
        else:
            return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


    def _intrinsic_delta(self, S, K, otype):
        """
        If T=0 (expiry), delta = 1 for call if S>K, 0 otherwise; 
                         = -1 for put if S<K, 0 otherwise.
        """
        if otype == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0


    def _execute_hedge_trade(self, timestamp, S_t, delta_to_trade):
        """
        Buys (if delta_to_trade > 0) or sells (if delta_to_trade < 0) that many shares of underlying.
        Updates self.hedged_shares, self.cash, and logs the trade.
        """
        # Cash outflow/inflow:
        #   If delta_to_trade > 0: we BUY shares → spend cash: delta_to_trade * S_t
        #   If delta_to_trade < 0: we SELL shares → receive cash: -(delta_to_trade * S_t)
        self.cash -= delta_to_trade * S_t
        self.hedged_shares += delta_to_trade

        # Log the hedge
        self.hedge_log.append({
            'timestamp': timestamp,
            'delta_before': -(self.hedged_shares - delta_to_trade),  # previous net‐delta before trade
            'delta_traded': delta_to_trade,
            'price': S_t
        })


    def _mark_to_market(self, timestamp, S_t, iv_dict):
        """
        Compute portfolio's mark‐to‐market equity:
          - Option PnL: for each open leg, (current mid_price – entry_price) × qty
          - Hedge PnL: current hedged_shares × S_t   (plus any entry/exit cash stored in self.cash)
        Returns a single float = total equity.
        """
        unrealized_option_pnl = 0.0

        for leg in self.open_legs:
            cid    = leg['contract_id']
            qty    = leg['quantity']
            strike = leg['strike']
            expiry = leg['expiry']
            entry  = leg['entry_price']
            otype  = leg['type']

            # Determine current price:
            if timestamp >= expiry:
                # intrinsic at expiry
                cur_price = max(0.0, (S_t - strike)) if otype == 'call' else max(0.0, (strike - S_t))
            else:
                T = max((expiry - timestamp).days, 0) / 252
                sigma = iv_dict.get(cid, None)
                if sigma is None:
                    # If we don't have this contract's IV, assume price = entry (i.e. zero PnL)
                    cur_price = entry
                else:
                    cur_price = self._bs_price(S_t, strike, T, sigma, otype)

            unrealized_option_pnl += qty * (cur_price - entry)

        hedge_mkt_value = self.hedged_shares * S_t
        total_equity = self.cash + unrealized_option_pnl + hedge_mkt_value
        return total_equity
    
    def get_day_bounds_unix_ms(date_str):
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        market_open = date.replace(hour=9,  minute=30, second=0, microsecond=0)
        market_close = date.replace(hour=16, minute=0,  second=0, microsecond=0)
        return int(market_open.timestamp() * 1000), int(market_close.timestamp() * 1000)


    # ─── OPTIONAL PROPERTY METHODS ────────────────────────────────────────────

    @property
    def current_timestamp(self):
        if 0 <= self.current_step < len(self.timestamps):
            return self.timestamps[self.current_step]
        else:
            return None

    @property
    def is_finished(self):
        return self.current_step >= len(self.timestamps)

    @property
    def equity_curve(self):
        """
        Returns a pd.DataFrame of self.timeline (with columns: timestamp, mtm_pnl, net_delta_pre_hedge, etc.)
        """
        return pd.DataFrame(self.timeline).set_index('timestamp')

    @property
    def hedge_trades(self):
        """
        Returns a pd.DataFrame of hedge trades logged.
        """
        return pd.DataFrame(self.hedge_log).set_index('timestamp')

    @staticmethod
    def parse_option_symbol(option_symbol: str):
        """
        Parse the option symbol to extract the underlying symbol, expiration date, option type, and strike price.

        Parameters
        ----------
        option_symbol : str
            The option symbol in the format 'O:<SYMBOL><YYMMDD><C/P><strike*1000>'.

        Returns
        -------
        dict
            A dictionary containing 'underlying_sym', 'expiration_date', 'option_type', and 'strike'.
        """
        # Parse underlying symbol (chars until first digit)
        first_digit_idx = re.search(r'\d', option_symbol).start()
        underlying_sym = option_symbol[2:first_digit_idx]

        # Parse expiration date (YYMMDD)
        exp_str = option_symbol[first_digit_idx:first_digit_idx+6]
        yy, mm, dd = int(exp_str[:2]), int(exp_str[2:4]), int(exp_str[4:6])
        expiration_date = datetime.datetime(2000 + yy, mm, dd)

        # Parse option type (C or P)
        opt_type_char = option_symbol[first_digit_idx+6]
        option_type = 'call' if opt_type_char == 'C' else 'put'

        # Parse strike (last digits → float dollars)
        strike_str = option_symbol[first_digit_idx+7:]
        strike = int(strike_str) / 1000.0

        return {
            'underlying_sym': underlying_sym,
            'expiration_date': expiration_date,
            'option_type': option_type,
            'strike': strike
        }
