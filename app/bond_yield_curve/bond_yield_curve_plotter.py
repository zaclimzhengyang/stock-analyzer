import pandas as pd
import plotly.graph_objs as go

from app.data.downloader import get_treasury_yield_curve
from constants import dates

curves = {label: get_treasury_yield_curve(date) for label, date in dates.items()}
df = pd.DataFrame(curves)

# 4. Plot interactive yield curves
fig = go.Figure()
for col in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=col))
fig.update_layout(
    title="US Treasury Yield Curves", xaxis_title="Maturity", yaxis_title="Yield (%)"
)
fig.show()
