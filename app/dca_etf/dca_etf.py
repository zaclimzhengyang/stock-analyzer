import datetime as dt
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
        if ticker not in all_prices.columns.get_level_values(0):
            return None

        df = all_prices[ticker]
        if "Adj Close" not in df.columns or df["Adj Close"].dropna().empty:
            return None

        prices = df["Adj Close"].dropna()

        # Only include ETFs that started trading before START_DATE
        if prices.index[0].date() > START_DATE:
            return None

        # Require at least 4 years of data
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
    """Download all tickers at once into a multi-index DataFrame."""
    df = yf.download(
        tickers,
        start=(START_DATE + dt.timedelta(days=-1)).strftime("%Y-%m-%d"),
        end=(END_DATE + dt.timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        group_by="ticker"
    )
    return df


def dcf_etf_main():
    start_time = dt.datetime.now()
    s = get_from_nyse_excel()
    s = s - set(TICKERS_TO_SKIP)
    print("Tickers to process:", len(s))

    # Download all at once
    all_prices = download_all_prices(list(s))

    # For testing purpose
    # s = list(s)[:500]
    # s = ["SMOG"]

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

    # Print summary table
    summary_df = top[["ticker", "invested", "final", "return", "cagr"]]
    print("\nSummary Results:")
    print(summary_df.to_string())
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSummary saved to {OUTPUT_CSV}")

    # Create plot
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
    end_time = dt.datetime.now()
    print("time taken:", end_time - start_time) # 0:00:38.018433


if __name__ == "__main__":
    main()
