"""Plotly forecast figures with a compatible offline Streamlit renderer."""

import json

import streamlit as st

from dashboard.data import METRICS


def forecast_spec(frame, metric: str) -> dict:
    title, unit = METRICS[metric]
    kind = "bar" if metric == "precipitation_mm" else "scatter"
    trace = {
        "type": kind, "x": [value.isoformat() for value in frame["valid_at_pkt"]],
        "y": frame["value"].tolist(), "text": frame["model"].tolist(),
        "marker": {"color": "#59e3ee"}, "name": title,
        "hovertemplate": "%{x|%d %b %H:%M} PKT<br>%{y:.2f} " + unit + "<br>%{text}<extra></extra>",
    }
    if kind == "scatter":
        trace.update(mode="markers" if metric == "wind_direction_degrees" else "lines+markers",
                     line={"color": "#59e3ee", "width": 2.5})
    axis = {"gridcolor": "#21384c", "zeroline": False}
    yaxis = {**axis, "title": {"text": unit}}
    if metric == "wind_direction_degrees":
        yaxis.update(range=[0, 360], tickvals=[0, 90, 180, 270, 360],
                     ticktext=["N · 0°", "E · 90°", "S · 180°", "W · 270°", "N · 360°"])
    if metric == "rain_occurred":
        yaxis["range"] = [0, 100]
    return {"data": [trace], "layout": {
        "height": 320, "margin": {"l": 50, "r": 20, "t": 16, "b": 45},
        "paper_bgcolor": "#102235", "plot_bgcolor": "#102235",
        "font": {"color": "#d8ebf3", "family": "system-ui"},
        "xaxis": {**axis, "type": "date", "tickformat": "%d %b %H:%M", "title": {"text": "Karachi time"}},
        "yaxis": yaxis,
    }}


def draw_forecast(frame, metric: str) -> None:
    figure = forecast_spec(frame, metric)
    options = {"displaylogo": False, "responsive": True}
    label = METRICS[metric][0] + " forecast"
    try:
        import plotly
    except ModuleNotFoundError:
        # Streamlit 1.67 bundles Plotly.js. A prebuilt JSON spec only needs its
        # chart message when the optional Python serializer is unavailable.
        from streamlit.elements.lib.layout_utils import LayoutConfig
        from streamlit.proto.PlotlyChart_pb2 import PlotlyChart

        proto = PlotlyChart(id=f"aetherix-{metric}", spec=json.dumps(figure),
                            config=json.dumps(options), alt=label)
        st._main._enqueue("plotly_chart", proto, layout_config=LayoutConfig(width="stretch", height=320))
    else:
        st.plotly_chart(figure, theme=None, config=options, width="stretch", alt=label)
