import pandas as pd
import math
import matplotlib.pyplot as plt

class Order:
    orders = []
    balance = 0.0
    balance_list = []

    def __init__(self, entry, tp, sl, units):
        self.entry = entry
        self.tp = tp
        self.sl = sl
        self.units = units
        self.active = True
        Order.orders.append(self)

    def validate(self, price):
        if not self.active:
            return

        # Long
        if self.units > 0:
            # hit TP
            if price >= self.tp:
                self.profit = self.units * (self.tp - self.entry)
                Order.balance += self.profit
                Order.balance_list.append(Order.balance)
                self.active = False
            # hit SL
            elif price <= self.sl:
                self.profit = self.units * (self.sl - self.entry)
                Order.balance += self.profit
                Order.balance_list.append(Order.balance)
                self.active = False

        # Short
        elif self.units < 0:
            # hit TP (price falls to TP)
            if price <= self.tp:
                self.profit = self.units * (self.tp - self.entry)
                Order.balance += self.profit
                Order.balance_list.append(Order.balance)
                self.active = False
            # hit SL (price rises to SL)
            elif price >= self.sl:
                self.profit = self.units * (self.sl - self.entry)
                Order.balance += self.profit
                Order.balance_list.append(Order.balance)
                self.active = False

    @staticmethod
    def validateAll(price):
        for o in Order.orders:
            o.validate(price)


def calc_mean(i, window, df):
    """
    Simple moving average of df['iv'] ending at index i.
    """
    if i < window - 1:
        return math.nan
    window_prices = df['iv'].iloc[i - window + 1 : i + 1]
    return window_prices.mean()


# --- parameters & data load ---
WINDOW_SIZE = 10
SL_DIST = 1.0          # fixed 1-point stop-loss
data0 = pd.read_csv("data/data0.csv", index_col=0)

# --- backtest loop ---
n = len(data0)
for i in range(WINDOW_SIZE, n):
    price = data0['iv'].iloc[i]
    mean  = calc_mean(i, WINDOW_SIZE, data0)

    # if we're above the mean, open a 1-unit short
    if price > mean:
        Order(entry=price,
              tp=mean,
              sl=price + SL_DIST,
              units=-1)

    # if we're below the mean, open a 1-unit long
    elif price < mean:
        Order(entry=price,
              tp=mean,
              sl=price - SL_DIST,
              units=1)

    # walk all open orders against today's price
    Order.validateAll(price)

# --- results ---
print(f"Final balance: {Order.balance:.2f}")

# --- plot ---
plt.figure(figsize=(10, 5))
plt.plot(Order.balance_list, label='Equity Curve')
plt.title('Mean Reversion Strategy - Balance Over Time')
plt.xlabel('Trades')
plt.ylabel('Cumulative P&L')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- calculate rolling mean of iv ---
data0['iv_mean'] = data0['iv'].rolling(WINDOW_SIZE).mean()

# --- plot iv and its mean ---
plt.figure(figsize=(12, 6))
plt.plot(data0['iv'], label='IV', alpha=0.8)
plt.plot(data0['iv_mean'], label=f'{WINDOW_SIZE}-Period Mean')
plt.title('IV and Rolling Mean')
plt.xlabel('Time')
plt.ylabel('IV')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
