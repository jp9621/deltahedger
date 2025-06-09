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
    
    All timestamps in this class are stored as integer milliseconds since epoch for consistency.
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
              - 'expiry' (as integer milliseconds since epoch), 
              - 'type'   ('call' or 'put').
        """
        self.underlying_prices = underlying_prices.copy()
        self.option_prices = option_prices.copy()
        
        self.underlying_prices.index = self.underlying_prices.index.astype(np.int64)
        
        if isinstance(self.option_prices.index, pd.MultiIndex):
            timestamps = self.option_prices.index.get_level_values(0).astype(np.int64)
            contract_ids = self.option_prices.index.get_level_values(1)
            self.option_prices.index = pd.MultiIndex.from_arrays([timestamps, contract_ids])
        
        if isinstance(self.option_prices['expiry'].iloc[0], (pd.Timestamp, datetime.datetime)):
            self.option_prices['expiry'] = self.option_prices['expiry'].astype(np.int64)

        self.underlying_prices = self.underlying_prices.sort_index()
        self.option_prices = self.option_prices.sort_index()

        self.r = r
        self.q = q

        self.delta_tolerance = delta_tolerance

        self.timestamps = list(self.underlying_prices.index)
        self.current_step = 0

        self.open_legs = []
        self.hedged_shares = 0.0
        self.cash = 0.0
        self.timeline = []
        self.hedge_log = []

    def forward(self):
        """
        Advance to the next timestamp. All timestamps are in milliseconds since epoch.
        """
        if self.current_step >= len(self.timestamps):
            raise IndexError("Already at end of price series.")

        t_ms = self.timestamps[self.current_step]
        S_t = self.underlying_prices.loc[t_ms]
        option_slice = self.option_prices.loc[t_ms]

        iv_dict = self._compute_implied_vols(option_slice, S_t)

        net_delta = 0.0
        for leg in self.open_legs:
            cid = leg['contract_id']
            qty = leg['quantity']
            strike = leg['strike']
            expiry_ms = leg['expiry']
            otype = leg['type']

            T = max((expiry_ms - t_ms) / (1000 * 3600 * 24 * 365), 0)

            if T <= 0:
                delta_i = self._intrinsic_delta(S_t, strike, otype)
            else:
                sigma = iv_dict.get(cid, None)
                if sigma is None:
                    delta_i = 0.0
                else:
                    delta_i = self._bs_delta(S_t, strike, T, sigma, otype)

            net_delta += qty * delta_i

        net_delta += self.hedged_shares

        delta_to_trade = 0.0
        if abs(net_delta) > self.delta_tolerance:
            delta_to_trade = -net_delta
            self._execute_hedge_trade(t_ms, S_t, delta_to_trade)

        net_delta_post = net_delta + delta_to_trade

        mtm_pnl = self._mark_to_market(t_ms, S_t, iv_dict)

        self.timeline.append({
            'timestamp': t_ms,
            'S_t': S_t,
            'net_delta_pre_hedge': net_delta,
            'net_delta_post_hedge': net_delta_post,
            'delta_traded': delta_to_trade,
            'hedged_shares': self.hedged_shares,
            'mtm_pnl': mtm_pnl
        })

        self.current_step += 1

    def open_position(self, timestamp_ms: int, contract_id: str, quantity: int):
        """
        Manually open a new option leg. timestamp_ms must be in milliseconds since epoch.
        """
        t_ms = self.timestamps[self.current_step]
        assert timestamp_ms == t_ms, "Can only open position at the current time index."

        row = self.option_prices.loc[(t_ms, contract_id)]
        strike, expiry_ms, otype, mid_price = (
            row['strike'], row['expiry'], row['type'], row['mid_price']
        )

        self.cash -= quantity * mid_price
        self.open_legs.append({
            'contract_id': contract_id,
            'type': otype,
            'strike': strike,
            'expiry': expiry_ms,
            'entry_price': mid_price,
            'quantity': quantity
        })

    def _compute_implied_vols(self, option_slice: pd.DataFrame, S_t: float) -> dict:
        """
        Compute implied volatilities. All timestamps are in milliseconds since epoch.
        """
        iv_dict = {}
        for cid, row in option_slice.iterrows():
            price = row['mid_price']
            K = row['strike']
            expiry_ms = row['expiry']
            otype = row['type']

            current_ms = self.timestamps[self.current_step]
            T = max((expiry_ms - current_ms) / (1000 * 3600 * 24 * 365), 0)

            if T <= 0:
                continue

            def f(sigma):
                return self._bs_price(S_t, K, T, sigma, otype) - price

            try:
                implied_vol = brentq(f, 1e-6, 5.0, maxiter=100, xtol=1e-6)
                iv_dict[cid] = implied_vol
            except (ValueError, RuntimeError):
                continue

        return iv_dict

    def _mark_to_market(self, timestamp_ms: int, S_t: float, iv_dict: dict) -> float:
        """
        Compute portfolio's mark‐to‐market equity. timestamp_ms is in milliseconds since epoch.
        """
        unrealized_option_pnl = 0.0

        for leg in self.open_legs:
            cid = leg['contract_id']
            qty = leg['quantity']
            strike = leg['strike']
            expiry_ms = leg['expiry']
            entry = leg['entry_price']
            otype = leg['type']

            if timestamp_ms >= expiry_ms:
                cur_price = max(0.0, (S_t - strike)) if otype == 'call' else max(0.0, (strike - S_t))
            else:
                T = max((expiry_ms - timestamp_ms) / (1000 * 3600 * 24 * 365), 0)
                sigma = iv_dict.get(cid, None)
                if sigma is None:
                    cur_price = entry
                else:
                    cur_price = self._bs_price(S_t, strike, T, sigma, otype)

            unrealized_option_pnl += qty * (cur_price - entry)

        hedge_mkt_value = self.hedged_shares * S_t
        total_equity = self.cash + unrealized_option_pnl + hedge_mkt_value
        return total_equity

    @staticmethod
    def to_ms_timestamp(dt) -> int:
        """Convert any datetime-like object to milliseconds since epoch."""
        if isinstance(dt, (int, np.int64)):
            return int(dt)
        elif isinstance(dt, (datetime.datetime, pd.Timestamp)):
            return int(dt.timestamp() * 1000)
        elif isinstance(dt, str):
            return int(pd.Timestamp(dt).timestamp() * 1000)
        else:
            raise ValueError(f"Cannot convert {type(dt)} to millisecond timestamp")

    def _execute_hedge_trade(self, timestamp, S_t, delta_to_trade):
        """
        Buys (if delta_to_trade > 0) or sells (if delta_to_trade < 0) that many shares of underlying.
        Updates self.hedged_shares, self.cash, and logs the trade.
        """
        self.cash -= delta_to_trade * S_t
        self.hedged_shares += delta_to_trade

        self.hedge_log.append({
            'timestamp': timestamp,
            'delta_before': -(self.hedged_shares - delta_to_trade),
            'delta_traded': delta_to_trade,
            'price': S_t
        })

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
        first_digit_idx = re.search(r'\d', option_symbol).start()
        underlying_sym = option_symbol[2:first_digit_idx]

        exp_str = option_symbol[first_digit_idx:first_digit_idx+6]
        yy, mm, dd = int(exp_str[:2]), int(exp_str[2:4]), int(exp_str[4:6])
        expiration_date = datetime.datetime(2000 + yy, mm, dd)

        opt_type_char = option_symbol[first_digit_idx+6]
        option_type = 'call' if opt_type_char == 'C' else 'put'

        strike_str = option_symbol[first_digit_idx+7:]
        strike = int(strike_str) / 1000.0

        return {
            'underlying_sym': underlying_sym,
            'expiration_date': expiration_date,
            'option_type': option_type,
            'strike': strike
        }
