from __future__ import annotations

import backtrader as bt


class MomentumStrategy(bt.Strategy):
    params = (("fast", 50), ("slow", 200),)

    def __init__(self):
        self.sma_fast = bt.ind.SMA(period=self.p.fast)
        self.sma_slow = bt.ind.SMA(period=self.p.slow)

    def next(self):
        if len(self.data) < max(self.p.fast, self.p.slow):
            return

        if not self.position:
            if self.sma_fast[0] > self.sma_slow[0]:
                self.buy()
        else:
            if self.sma_fast[0] < self.sma_slow[0]:
                self.sell()


class MeanReversionStrategy(bt.Strategy):
    params = (("rsi_buy", 30), ("rsi_sell", 70), ("rsi_period", 14))

    def __init__(self):
        self.rsi = bt.ind.RSI(period=self.p.rsi_period)

    def next(self):
        if len(self.data) < self.rsi.p.period:
            return

        if not self.position:
            if self.rsi[0] < self.p.rsi_buy:
                self.buy()
        else:
            if self.rsi[0] > self.p.rsi_sell:
                self.sell()
