import pandas as pd
import plotly.graph_objs as go
from pandas_datareader import data as pdr

# 2. Define FRED series for US Treasury yields
series = {
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
    "30Y": "DGS30"
}

# 3. Download yield data for current and historical dates
def get_yield_curve(date):
    yields = {}
    for label, code in series.items():
        try:
            val = pdr.DataReader(code, "fred", date, date).iloc[0, 0]
            yields[label] = val
        except Exception:
            yields[label] = None
    return yields

dates = {
    "Current": pd.Timestamp.today().strftime("%Y-%m-%d"),
    "2008": "2008-09-15",
    "2020": "2020-03-23"
}

curves = {label: get_yield_curve(date) for label, date in dates.items()}
df = pd.DataFrame(curves)

# 4. Plot interactive yield curves
fig = go.Figure()
for col in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines+markers', name=col))
fig.update_layout(title="US Treasury Yield Curves", xaxis_title="Maturity", yaxis_title="Yield (%)")
fig.show()

# 5. Short write-up (markdown cell)
# ...existing code...