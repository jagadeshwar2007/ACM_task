from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

t6 = Path(__file__).resolve().parent.parent / "task_06_housing_price_predictor"


@st.cache_resource
def load_model():
    return joblib.load(t6 / "model" / "housing_model.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(t6 / "data" / "housing.csv")


model = load_model()
data = load_data()

st.title("California house price predictor")
st.write("Change the values on the left and the prediction updates.")

# inputs
st.sidebar.header("House details")
income = st.sidebar.slider("Median income (x $10,000)", 0.5, 15.0, 4.0, 0.1)
age = st.sidebar.slider("House age (years)", 1, 52, 28)
rooms = st.sidebar.slider("Total rooms", 100, 12000, 2600, 100)
bedrooms = st.sidebar.slider("Total bedrooms", 20, 2500, 530, 10)
population = st.sidebar.slider("Population", 100, 8000, 1400, 50)
households = st.sidebar.slider("Households", 20, 2500, 500, 10)
latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 37.77, 0.01)
longitude = st.sidebar.slider("Longitude", -124.5, -114.3, -122.42, 0.01)
proximity = st.sidebar.selectbox("Ocean proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

house = pd.DataFrame([{
    "longitude": longitude, "latitude": latitude, "housing_median_age": age,
    "total_rooms": rooms, "total_bedrooms": bedrooms, "population": population,
    "households": households, "median_income": income, "ocean_proximity": proximity,
}])
price = model.predict(house)[0]

st.metric("Predicted house value", f"${price:,.0f}")

# chart 1: where the prediction sits compared to all houses
fig1 = px.histogram(data, x="median_house_value", nbins=50, title="Your prediction vs all districts")
fig1.add_vline(x=price, line_color="red", line_width=3)
st.plotly_chart(fig1)

# chart 2: how price changes with income (other inputs stay the same)
incomes = [i / 2 for i in range(1, 31)]
sweep = pd.concat([house.assign(median_income=i) for i in incomes])
sweep["predicted_price"] = model.predict(sweep)
fig2 = px.line(sweep, x="median_income", y="predicted_price", title="Price vs median income")
fig2.add_scatter(x=[income], y=[price], mode="markers", marker=dict(size=12, color="red"), showlegend=False)
st.plotly_chart(fig2)
