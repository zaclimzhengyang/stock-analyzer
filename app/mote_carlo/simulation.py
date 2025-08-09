import numpy as np
import pandas as pd
import datetime

from app.data import downloader


def mc_var(returns: pd.Series, alpha=5):
    """Input: pandas series of returns
    Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")


def mc_cvar(returns: pd.Series, alpha=5):
    """Input: pandas series of returns
    Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        below_var = returns <= mc_var(returns, alpha=alpha)
        return returns[below_var].mean()
    else:
        raise TypeError("Expected a pandas data series.")


def mc_simulation(ticker: str) -> dict:
    end_date: datetime = datetime.datetime.now()
    start_date: datetime = end_date - datetime.timedelta(days=300)

    mean_returns, cov_matrix = downloader.get_mean_returns_cov_matrix(
        ticker, start_date, end_date
    )

    weights: np.ndarray = np.random.random(len(mean_returns))
    weights /= np.sum(weights)

    # Monte Carlo Method
    mc_sims = 400  # number of simulations
    T = 100  # timeframe in days
    initialPortfolio: int = 10000

    mean_m: np.ndarray = np.full(shape=(T, len(weights)), fill_value=mean_returns).T
    portfolio_sims: np.ndarray = np.full(shape=(T, mc_sims), fill_value=0.0)

    for m in range(mc_sims):
        Z: np.ndarray = np.random.normal(size=(T, len(weights)))
        L: np.ndarray = np.linalg.cholesky(cov_matrix)
        daily_returns: np.ndarray = mean_m + np.inner(L, Z)
        portfolio_sims[:, m] = (
            np.cumprod(np.inner(weights, daily_returns.T) + 1) * initialPortfolio
        )

    port_results: pd.Series = pd.Series(portfolio_sims[-1, :])

    var: float = initialPortfolio - mc_var(port_results, alpha=5)
    cvar: float = initialPortfolio - mc_cvar(port_results, alpha=5)

    return {
        "simulations": portfolio_sims.tolist(),
        "VaR_5": round(var, 2),
        "CVaR_5": round(cvar, 2),
    }
