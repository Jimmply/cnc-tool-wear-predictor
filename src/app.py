"""CNC Tool Wear Prediction Dashboard — Streamlit app."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import FEATURE_COLS, WEAR_STATE_COLORS, ToolWearGenerator
from predictor import WEAR_ORDER, ToolWearPredictor

st.set_page_config(
    page_title="CNC Tool Wear Monitor",
    page_icon="⚙️",
    layout="wide",
)


# ------------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------------

@st.cache_resource
def load_model(n_tools: int) -> tuple[pd.DataFrame, ToolWearPredictor, object]:
    df = ToolWearGenerator(n_tools=n_tools).generate()
    predictor = ToolWearPredictor()
    results = predictor.fit(df)
    return df, predictor, results


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

st.sidebar.title("⚙️ CNC Tool Wear Monitor")
st.sidebar.markdown("Real-time wear state classification and RUL prediction for a milling machine fleet.")

n_tools = st.sidebar.slider("Fleet size (tools)", 5, 40, 20, step=5)
df, predictor, results = load_model(n_tools)

all_tools = sorted(df["tool_id"].unique())
selected_tool = st.sidebar.selectbox("Inspect tool", all_tools)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model RUL MAE:** {results.rul_mae:.0f} cycles")
st.sidebar.markdown(f"**Total records:** {len(df):,}")

# ------------------------------------------------------------------
# Fleet overview
# ------------------------------------------------------------------

fleet = predictor.latest_per_tool(df)

st.title("CNC Tool Wear Prediction Dashboard")

col1, col2, col3 = st.columns(3)
n_fresh    = (fleet["predicted_state"] == "Fresh").sum()
n_worn     = (fleet["predicted_state"] == "Worn").sum()
n_critical = (fleet["predicted_state"] == "Critical").sum()

col1.metric("🟢 Fresh",    n_fresh)
col2.metric("🟡 Worn",     n_worn)
col3.metric("🔴 Critical", n_critical)

st.markdown("---")

# Fleet scatter: cut_number vs wear index, coloured by state
latest_with_data = fleet.merge(
    df[["tool_id", "cut_number", "wear_index"]],
    on=["tool_id", "cut_number"],
)

fig_fleet = px.scatter(
    latest_with_data,
    x="cut_number",
    y="wear_index",
    color="predicted_state",
    color_discrete_map=WEAR_STATE_COLORS,
    hover_data=["tool_id", "predicted_rul"],
    title="Fleet Wear State at Latest Cut",
    labels={"cut_number": "Total Cuts Made", "wear_index": "Wear Index (0→1)"},
    category_orders={"predicted_state": WEAR_ORDER},
)
fig_fleet.update_traces(marker_size=10)
st.plotly_chart(fig_fleet, use_container_width=True)

# ------------------------------------------------------------------
# Selected tool detail
# ------------------------------------------------------------------

st.subheader(f"Tool Detail — {selected_tool}")
tool_df = df[df["tool_id"] == selected_tool].copy()
tool_preds = predictor.predict(tool_df)

# Current state card
last = tool_preds.iloc[-1]
state = last["predicted_state"]
rul   = int(last["predicted_rul"])
state_color = WEAR_STATE_COLORS[state]

c1, c2, c3 = st.columns(3)
c1.markdown(
    f"<div style='background:{state_color};padding:12px;border-radius:8px;"
    f"text-align:center;color:white;font-size:18px;font-weight:bold'>"
    f"Wear State<br>{state}</div>",
    unsafe_allow_html=True,
)
c2.metric("Estimated RUL", f"{rul} cuts")
c3.metric("Cuts completed", f"{int(last['cut_number'])}")

st.markdown("")

# Sensor trend charts — 2×3 grid
sensor_labels = {
    "vibration_rms_mm_s": "Vibration RMS (mm/s)",
    "vibration_kurtosis": "Vibration Kurtosis",
    "spindle_current_a": "Spindle Current (A)",
    "acoustic_emission_db": "Acoustic Emission (dB)",
    "cutting_force_n": "Cutting Force (N)",
    "surface_roughness_um": "Surface Roughness Ra (μm)",
}

cols = st.columns(3)
for i, (col, label) in enumerate(sensor_labels.items()):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tool_df["cut_number"],
        y=tool_df[col],
        mode="lines",
        line=dict(width=1.5, color="#3498db"),
        name=label,
    ))
    fig.update_layout(
        title=label,
        xaxis_title="Cut #",
        yaxis_title="",
        height=220,
        margin=dict(l=30, r=10, t=35, b=30),
        showlegend=False,
    )
    cols[i % 3].plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------------

st.subheader("Feature Importance (Wear State Classifier)")
fi = results.feature_importances
fig_fi = px.bar(
    fi,
    orientation="h",
    labels={"value": "Importance", "index": "Feature"},
    color=fi.values,
    color_continuous_scale="Blues",
)
fig_fi.update_layout(showlegend=False, coloraxis_showscale=False, height=280)
st.plotly_chart(fig_fi, use_container_width=True)

# ------------------------------------------------------------------
# Fleet table
# ------------------------------------------------------------------

st.subheader("Fleet Status Table")
display_fleet = fleet[
    ["tool_id", "predicted_state", "predicted_rul", "cut_number",
     "vibration_rms_mm_s", "spindle_current_a", "surface_roughness_um"]
].rename(columns={
    "tool_id": "Tool",
    "predicted_state": "State",
    "predicted_rul": "RUL (cuts)",
    "cut_number": "Cuts Made",
    "vibration_rms_mm_s": "Vib RMS",
    "spindle_current_a": "Spindle A",
    "surface_roughness_um": "Ra (μm)",
}).sort_values("RUL (cuts)")

def _color_state(val: str) -> str:
    colors = {"Fresh": "color: #27ae60", "Worn": "color: #e67e22", "Critical": "color: #c0392b; font-weight:bold"}
    return colors.get(val, "")

st.dataframe(
    display_fleet.style.applymap(_color_state, subset=["State"]),
    use_container_width=True,
    hide_index=True,
)
