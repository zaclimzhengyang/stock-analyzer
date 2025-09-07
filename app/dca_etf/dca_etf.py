import datetime as dt
import os
import pickle
import glob

import pandas as pd
import numpy as np
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from tqdm import tqdm

START_DATE = dt.date(2020, 1, 1)
END_DATE = dt.date(2025, 8, 30)
MONTHLY_INVEST = 100.0
OUTPUT_CSV = "top10_etfs_combined.csv"
OUTPUT_PNG = "top10_etfs_combined.png"
TICKERS_TO_SKIP = ["OND", "IBCA"]  # Suspected to have bad data
CACHE_CHUNK_SIZE = 100  # number of tickers per chunk
CACHE_PATTERN = "all_prices_cache_{}.pkl"  # pattern for chunked cache files

# === Helper Functions ===

def get_from_nyse_excel():
    url = "https://www.nyse.com/publicdocs/nyse/markets/nyse-arca/NYSE_Arca_Equities_LMM_Current.xlsx"
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_excel(r.content)

    col_keys = [c for c in df.columns if "symbol" in str(c).lower()]
    if not col_keys:
        return set()
    symbols = df[col_keys[0]].astype(str).str.upper().tolist()
    clean = {
        s.strip().replace(".", "-")
        for s in symbols if s and len(s) <= 8 and s.isalnum()
    }
    print("NYSE Excel:", len(clean), "tickers found")
    return clean


def first_trading_days(prices):
    dates = []
    cur = dt.date(START_DATE.year, START_DATE.month, 1)
    while cur <= END_DATE:
        idx = prices.index.searchsorted(pd.Timestamp(cur))
        if idx < len(prices):
            ts = prices.index[idx]
            if ts.date() <= END_DATE:
                dates.append(ts)
        cur = (cur + pd.offsets.MonthBegin(1)).date()
    return dates


def run_dca_from_prices(ticker, all_prices):
    try:
        if ticker not in all_prices:
            return None

        df = all_prices[ticker]
        if "Adj Close" not in df.columns or df["Adj Close"].dropna().empty:
            return None

        prices = df["Adj Close"].dropna()

        if prices.index[0].date() > START_DATE:
            return None
        if len(prices) < 48:
            return None

        ftd = first_trading_days(prices)
        if len(ftd) < 12:
            return None

        shares = invested = 0.0
        vals, dates = [], []
        for d in ftd:
            p = float(prices.loc[d])
            shares += MONTHLY_INVEST / p
            invested += MONTHLY_INVEST
            vals.append(shares * p)
            dates.append(d)

        final_value = float(shares * prices.iloc[-1])
        return {
            "ticker": ticker,
            "invested": invested,
            "final": final_value,
            "series": pd.Series(vals, index=dates),
        }
    except Exception as e:
        print(f"Failed {ticker}: {e}")
        return None


def download_all_prices(tickers):
    df = yf.download(
        tickers,
        start=(START_DATE + dt.timedelta(days=-1)).strftime("%Y-%m-%d"),
        end=(END_DATE + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        group_by="ticker"
    )
    out = {}
    for t in tickers:
        if t in df.columns.get_level_values(0):
            out[t] = df[t].dropna()
    return out


def save_cache(all_prices: dict):
    """Split all_prices into chunks and save as multiple pickle files."""
    tickers = sorted(all_prices.keys())
    for i in range(0, len(tickers), CACHE_CHUNK_SIZE):
        chunk_tickers = tickers[i:i + CACHE_CHUNK_SIZE]
        chunk_data = {t: all_prices[t] for t in chunk_tickers}
        filename = CACHE_PATTERN.format(i // CACHE_CHUNK_SIZE + 1)
        with open(filename, "wb") as f:
            pickle.dump(chunk_data, f)
        print(f"Saved {filename} with {len(chunk_tickers)} tickers.")


def load_cache() -> dict:
    """Load all pickle chunks and merge into one dictionary."""
    all_prices = {}
    files = sorted(glob.glob(CACHE_PATTERN.format("*")))
    if not files:
        return {}
    for f in files:
        with open(f, "rb") as fh:
            chunk = pickle.load(fh)
            all_prices.update(chunk)
        print(f"Loaded {f} with {len(chunk)} tickers.")
    print(f"Total tickers loaded: {len(all_prices)}")
    return all_prices


def dcf_etf_main():
    start_time = dt.datetime.now()
    s = get_from_nyse_excel()
    s = s - set(TICKERS_TO_SKIP)
    tickers = sorted(list(s))
    print("Tickers to process:", len(tickers))

    # Load or download
    if glob.glob(CACHE_PATTERN.format("*")):
        all_prices = load_cache()
    else:
        all_prices = download_all_prices(tickers)
        save_cache(all_prices)

    # Run DCA
    results = []
    for t in tqdm(sorted(s), desc="DCA backtests"):
        res = run_dca_from_prices(t, all_prices)
        if res:
            total = res["invested"]
            fin = res["final"]
            ret = fin / total - 1
            m = len(res["series"])
            cagr = (fin / total) ** (12 / m) - 1 if m else np.nan
            results.append({
                "ticker": t,
                "invested": float(total),
                "final": float(fin),
                "return": float(ret),
                "cagr": float(cagr) if not np.isnan(cagr) else np.nan,
                "series": res["series"]
            })

    if not results:
        print("No results; maybe no ticker had full data.")
        return

    df = pd.DataFrame(results).sort_values("return", ascending=False).reset_index(drop=True)
    top = df.head(10)
    summary_df = top[["ticker", "invested", "final", "return", "cagr"]]
    print("\nSummary Results:")
    print(summary_df.to_string())
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSummary saved to {OUTPUT_CSV}")

    plt.figure(figsize=(14, 8))
    for _, row in top.iterrows():
        srs = row["series"]
        invested = row["invested"]
        final_value = row["final"]
        plt.plot(
            srs.index,
            srs.values,
            label=f"{row['ticker']} | Invested: ${invested:,.0f} | Final: ${final_value:,.0f}"
        )

    plt.title("Top 10 ETF DCA Performance (absolute $)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend(fontsize="small")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.show()
    print("Time taken:", dt.datetime.now() - start_time)


if __name__ == "__main__":
    dcf_etf_main()
