import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime
import yfinance as yf

# Function to load data
@st.cache_data
def load_data():
    data = yf.download("GHS=X", start="2010-01-01", end=datetime.now().strftime("%Y-%m-%d"))
    data['Date'] = data.index
    data.reset_index(drop=True, inplace=True)
    return data

# Function to get the current exchange rate
def get_current_rate():
    current_data = yf.download("GHS=X", period="1d")
    current_rate = current_data['Close'].iloc[-1]
    return current_rate

# Function to train model
@st.cache_resource
def train_model(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = data['Date'].map(datetime.toordinal)
    X = data[['Date']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Load and display data
data = load_data()
st.title("Exchange Rate Predictor: Ghana Cedis (GHS) to US Dollars (USD)")
st.write("## Historical Data")
st.line_chart(data['Close'])

# Get the current exchange rate
current_rate = get_current_rate()
st.write(f"### Current Exchange Rate (GHS to USD): {current_rate:.2f}")

# Train model
model = train_model(data)

# User input for prediction
st.write("## Predict Future Exchange Rate")
date_input = st.date_input("Select a date", datetime.now())
date_ordinal = datetime.toordinal(date_input)
prediction = model.predict([[date_ordinal]])

st.write(f"### Predicted Exchange Rate (GHS to USD) for {date_input}: {prediction[0]:.2f}")

# Plot predictions
fig, ax = plt.subplots()
ax.plot(data['Date'], data['Close'], label="Historical")
ax.axvline(x=date_input, color='red', linestyle='--', label="Prediction Date")
ax.scatter([date_input], [prediction], color='red', label="Prediction")
ax.legend()
st.pyplot(fig)
