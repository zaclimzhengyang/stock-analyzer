import pandas as pd

treasury_fred_series = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

dates = {
    "Current": pd.Timestamp.today().strftime("%Y-%m-%d"),
    "2008": "2008-09-15",
    "2020": "2020-03-23",
    "2023": "2023-03-01",
    "2024": "2024-01-01",
    "2025": "2025-06-01",
}
