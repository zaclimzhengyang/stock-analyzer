import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest, skew, kurtosis
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde

from app.data.downloader import get_price_data


def pdf(ticker: str, start_date: str, end_date: str):
    """
    Plot empirical PDF of daily log-returns vs. fitted normal PDF.

    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL').
    - start_date: Start date for historical data (YYYY-MM-DD).
    - end_date: End date for historical data (YYYY-MM-DD).

    Outputs:
    - Saves a PNG and PDF file with the plot.
    - Returns mean, std, skewness, kurtosis, KS test results.
    """
    # Download daily close prices
    df = get_price_data(ticker, start_date, end_date)

    # Compute log returns (or use simple pct_change)
    df['log_ret'] = np.log(df['Close']).diff().dropna()
    data = df['log_ret'].dropna().values

    # Fit empirical stats
    mu = data.mean()
    sigma = data.std(ddof=1)

    # Theoretical normal PDF (fitted using sample mean/std)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    theoretical_pdf = norm.pdf(x, mu, sigma)

    # Empirical histogram (density)
    count, bins = np.histogram(data, bins=50, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Optional: overlay Gaussian kernel density estimate (KDE) via simple method
    # (We avoid seaborn as per instructions; we can use scipy gaussian_kde if desired.)
    kde = gaussian_kde(data)
    kde_vals = kde(x)

    # Diagnostics
    ks_stat, ks_p = kstest((data - mu) / sigma, 'norm')
    s = skew(data)
    k = kurtosis(data, fisher=True)

    # Create figure (instead of saving)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bin_centers, count, width=(bins[1] - bins[0]), alpha=0.45, label='Empirical (histogram)')
    ax.plot(x, kde_vals, lw=2, label='KDE (empirical)', linestyle='--')
    ax.plot(x, theoretical_pdf, lw=2.5, label=f'Normal PDF (μ={mu:.4g}, σ={sigma:.4g})')
    ax.set_xlabel('Log return')
    ax.set_ylabel('Density')
    ax.set_title(f'{ticker} Daily Log-Returns: Empirical PDF vs. Fitted Normal PDF\n{start_date} to {end_date}')
    ax.legend()
    text = (
        f"N = {len(data)}\nMean = {mu:.6f}\nStd = {sigma:.6f}\nSkew = {s:.4f}\nEx. kurtosis = {k:.4f}\n"
        f"KS stat = {ks_stat:.4f}, p = {ks_p:.4f}"
    )
    ax.text(0.98, 0.98, text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    return fig, {
        "Mean": mu,
        "Standard Deviation": sigma,
        "Skewness": s,
        "Kurtosis": k,
        "KS Stats": ks_stat,
        "KS P Value": ks_p
    }
