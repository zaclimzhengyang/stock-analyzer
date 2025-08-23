from __future__ import annotations

import math
import os
from typing import Tuple

import backtrader as bt
import pandas as pd
import numpy as np

from app.back_trader.models import RunSettings, ReturnsAnalyzer
from app.back_trader.strategies import MomentumStrategy, MeanReversionStrategy
from app.data.downloader import download_data


def _compute_metrics_from_equity(equity: pd.Series, trading_days: int = 252) -> Tuple[float, float, float, float]:
    """Given equity curve (indexed by datetime), return (total_ret, CAGR, vol, sharpe, max_dd)."""
    equity = equity.dropna()
    if equity.empty:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    start_val = equity.iloc[0]
    end_val = equity.iloc[-1]
    total_ret = end_val / start_val - 1.0

    # Daily returns
    rets = equity.pct_change().dropna()
    vol = rets.std() * math.sqrt(trading_days) if len(rets) > 1 else np.nan
    mean_daily = rets.mean()
    sharpe = (
        (mean_daily / rets.std() * math.sqrt(trading_days))
        if rets.std() > 0
        else np.nan
    )

    # CAGR
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cagr = (end_val / start_val) ** (1 / years) - 1 if years and years > 0 else np.nan

    # Max Drawdown
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = dd.min() if not dd.empty else np.nan

    return total_ret, cagr, vol, sharpe, max_dd


def run_backtest(strategy, df: pd.DataFrame, settings: RunSettings):
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(settings.cash)
    cerebro.broker.setcommission(commission=settings.commission)
    cerebro.broker.set_slippage_perc(perc=settings.slippage)

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    # Add strategy with correct parameters
    if strategy == MomentumStrategy:
        cerebro.addstrategy(strategy, fast=settings.fast, slow=settings.slow)
    elif strategy == MeanReversionStrategy:
        cerebro.addstrategy(strategy, rsi_buy=settings.rsi_buy, rsi_sell=settings.rsi_sell)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(ReturnsAnalyzer, _name="equity")

    fast = settings.fast
    slow = settings.slow
    min_period = max(fast, slow)
    if len(df) < min_period:
        raise ValueError(f"Not enough data: need at least {min_period} rows, got {len(df)}")

    results = cerebro.run()
    strat = results[0]

    # Get portfolio value
    port_value = cerebro.broker.getvalue()

    # CAGR calculation
    cagr = (port_value / settings.cash) ** (
            1 / ((pd.to_datetime(settings.end) - pd.to_datetime(settings.start)).days / 365.25)) - 1

    # Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()

    return {
        "final_value": port_value,
        "CAGR": cagr,
        "MaxDrawdown": drawdown["max"]["drawdown"]
    }


def save_results(results: dict, settings: RunSettings):
    os.makedirs(settings.out, exist_ok=True)
    metrics_path = os.path.join(settings.out, "metrics.csv")
    pd.DataFrame(results).T.to_csv(metrics_path)
    print(f"Results saved to {metrics_path}")


def backtrader_analyze(settings: RunSettings):
    print("Running backtest with settings:")
    print(settings)

    all_results = {}

    for ticker in settings.tickers:
        print(f"\n=== {ticker} ===")
        df = download_data(ticker, settings.start, settings.end)

        # Momentum
        mom_results = run_backtest(MomentumStrategy, df, settings)

        # Mean Reversion
        meanrev_results = run_backtest(MeanReversionStrategy, df, settings)

        all_results[ticker] = {
            "Momentum_FinalValue": mom_results["final_value"],
            "Momentum_CAGR": mom_results["CAGR"],
            "MeanRev_FinalValue": meanrev_results["final_value"],
            "MeanRev_CAGR": meanrev_results["CAGR"],
        }

    save_results(all_results, settings)
    return all_results