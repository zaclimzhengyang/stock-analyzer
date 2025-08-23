from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import backtrader as bt


@dataclass
class RunSettings:
    tickers: list[str]
    start: str
    end: Optional[str]
    cash: float
    commission: float
    slippage: float
    fast: int
    slow: int
    rsi_buy: int
    rsi_sell: int
    out: str



@dataclass
class Metrics:
    ticker: str
    strategy: str
    start: datetime
    end: datetime
    days: int
    years: float
    start_value: float
    end_value: float
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    max_drawdown: float


class ReturnsAnalyzer(bt.Analyzer):
    """Collect daily strategy returns based on portfolio value."""

    def start(self):
        self.values = []

    def next(self):
        self.values.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        return self.values
