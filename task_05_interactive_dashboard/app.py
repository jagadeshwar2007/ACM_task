"""Task 5 — Interactive AI dashboard (Streamlit) for the California housing price model.

Run:  streamlit run app.py
Every widget change re-runs the model and updates the charts in real time.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
# Trained pipeline + dataset come from Task 6 (one shared copy in the repo)
T6 = BASE.parent / "task_06_housing_price_predictor"
MODEL_DIR = T6 / "model" if (T6 / "model").exists() else BASE / "model"
DATA_PATH = T6 / "data" / "housing.csv" if (T6 / "data").exists() else BASE / "housing.csv"
st.set_page_config(page_title="California Housing Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "housing_model.joblib")


@st.cache_data
def load_meta():
    return json.loads((MODEL_DIR / "model_metadata.json").read_text())


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


model, meta, df = load_model(), load_meta(), load_data()
median_all = df["median_house_value"].median()

# ---------------------------------------------------------------- sidebar inputs
st.sidebar.header("Describe the district")
income = st.sidebar.slider("Median income ($10,000s)", 0.5, 15.0, 4.0, 0.1)
age = st.sidebar.slider("Median house age (years)", 1, 52, 28)
rooms = st.sidebar.slider("Total rooms", 100, 12000, 2600, 100)
bedrooms = st.sidebar.slider("Total bedrooms", 20, 2500, 530, 10)
population = st.sidebar.slider("Population", 100, 8000, 1400, 50)
households = st.sidebar.slider("Households", 20, 2500, 500, 10)
proximity = st.sidebar.selectbox("Ocean proximity", meta["ocean_proximity_values"], index=0)

# Location: pick a preset city or fine-tune coordinates
CITIES = {
    "San Francisco": (37.77, -122.42), "Los Angeles": (34.05, -118.25), "San Diego": (32.72, -117.16),
    "San Jose": (37.34, -121.89), "Sacramento": (38.58, -121.49), "Fresno": (36.74, -119.79),
    "Custom": None,
}
city = st.sidebar.selectbox("Location preset", list(CITIES))
lat0, lon0 = CITIES[city] or (36.5, -119.5)
lat = st.sidebar.slider("Latitude", 32.5, 42.0, float(lat0), 0.01)
lon = st.sidebar.slider("Longitude", -124.5, -114.3, float(lon0), 0.01)

row = pd.DataFrame([{
    "longitude": lon, "latitude": lat, "housing_median_age": age, "total_rooms": rooms,
    "total_bedrooms": bedrooms, "population": population, "households": households,
    "median_income": income, "ocean_proximity": proximity,
}])
pred = float(model.predict(row)[0])

# ---------------------------------------------------------------- header + KPIs
st.title("🏠 California Housing Price Predictor")
st.caption("Random Forest pipeline (median imputation → StandardScaler → one-hot → forest). "
           "Move any slider and the prediction and charts update instantly.")

c1, c2, c3 = st.columns(3)
c1.metric("Predicted median house value", f"${pred:,.0f}")
c2.metric("vs. California median", f"${pred - median_all:+,.0f}", f"{(pred / median_all - 1) * 100:+.0f}%")
c3.metric("Model test R² / RMSE", f"{meta['test_metrics']['R2']:.2f}", f"RMSE ≈ ${meta['test_metrics']['RMSE']:,.0f}",
          delta_color="off")
if pred >= 490_000:
    st.warning("The training data is capped at $500,001, so predictions near this value are a lower bound.")

# ---------------------------------------------------------------- charts
left, right = st.columns(2)

with left:
    st.subheader("Where does this fall in the market?")
    fig = px.histogram(df, x="median_house_value", nbins=60, opacity=0.75,
                       labels={"median_house_value": "Median house value ($)"})
    fig.add_vline(x=pred, line_color="crimson", line_width=3,
                  annotation_text=f"Prediction ${pred:,.0f}", annotation_position="top")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, yaxis_title="Districts")
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Price vs. median income (live sweep)")
    incomes = [round(0.5 + i * 0.5, 1) for i in range(30)]
    sweep = pd.concat([row.assign(median_income=i) for i in incomes], ignore_index=True)
    sweep["predicted"] = model.predict(sweep)
    fig2 = px.line(sweep, x="median_income", y="predicted", markers=True,
                   labels={"median_income": "Median income ($10,000s)", "predicted": "Predicted value ($)"})
    fig2.add_trace(go.Scatter(x=[income], y=[pred], mode="markers", marker=dict(size=14, color="crimson"),
                              name="Your input"))
    fig2.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    st.plotly_chart(fig2, width="stretch")

left2, right2 = st.columns(2)

with left2:
    st.subheader("Your location on the price map")
    sample = df.sample(2500, random_state=0)
    fig3 = px.scatter(sample, x="longitude", y="latitude", color="median_house_value", opacity=0.4,
                      color_continuous_scale="Viridis", labels={"median_house_value": "Value ($)"})
    fig3.add_trace(go.Scatter(x=[lon], y=[lat], mode="markers", name="Your input",
                              marker=dict(size=28, color="red", symbol="star", line=dict(color="black", width=2))))
    fig3.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig3, width="stretch")

with right2:
    st.subheader("What drives the model overall?")
    imp = pd.Series(meta["feature_importances"]).sort_values().tail(8)
    fig4 = px.bar(imp, orientation="h", labels={"value": "Importance", "index": ""})
    fig4.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    st.plotly_chart(fig4, width="stretch")

with st.expander("See the exact input sent to the model"):
    st.dataframe(row, hide_index=True)
