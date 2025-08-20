import numpy as np
import pandas as pd
import datetime

from app.data import downloader


def mc_var(returns: pd.Series, alpha=5):
    """
    Calculate Value at Risk (VaR) at the given percentile.

    Financial Description:
    - VaR: The maximum expected loss at a given confidence level (e.g., 5%).
    """
    return np.percentile(returns, alpha)


def mc_cvar(returns: pd.Series, alpha=5):
    """
    Calculate Conditional Value at Risk (CVaR) at the given percentile.

    Financial Description:
    - CVaR: The expected loss in the worst-case (tail) scenarios beyond the VaR threshold.
    """
    var = mc_var(returns, alpha=alpha)
    return returns[returns <= var].mean()


def mc_simulation(ticker: str) -> dict:
    """
    Run a Monte Carlo simulation for a given ticker.

    Simulates multiple future price paths using historical mean returns and covariance,
    and computes portfolio Value at Risk (VaR) and Conditional Value at Risk (CVaR).

    Financial Description:
    - Monte Carlo simulation: Randomly generates price paths to estimate risk.
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=300)

    mean_returns, cov_matrix = downloader.get_mean_returns_cov_matrix(
        ticker, start_date, end_date
    )

    weights = np.random.random(len(mean_returns))
    weights /= np.sum(weights)

    mc_sims = 400
    T = 100
    initial_portfolio = 10000

    mean_matrix = np.tile(mean_returns.values, (T, 1))
    portfolio_sims = np.zeros((T, mc_sims))

    chol_matrix = np.linalg.cholesky(cov_matrix)

    for m in range(mc_sims):
        Z = np.random.normal(size=(T, len(weights)))
        daily_returns = mean_matrix + Z @ chol_matrix.T
        portfolio_path = np.cumprod(1 + daily_returns @ weights) * initial_portfolio
        portfolio_sims[:, m] = portfolio_path

    port_results = pd.Series(portfolio_sims[-1, :])

    var = initial_portfolio - mc_var(port_results, alpha=5)
    cvar = initial_portfolio - mc_cvar(port_results, alpha=5)

    return {
        "simulations": portfolio_sims.tolist(),
        "VaR_5": round(var, 2),
        "CVaR_5": round(cvar, 2),
    }
