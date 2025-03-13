import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# ---- Cache the Live Bitcoin Price ----
@st.cache_data(ttl=60*5)  # Cache for 5 minutes (300 seconds)
def get_current_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data["bitcoin"]["usd"]  # Extract the current price

# ---- Load Data ----
try:
    # Load Bitcoin Price Data
    df_prices = pd.read_csv("bitcoin_prices.csv", parse_dates=["Date"], index_col="Date")

    # Load Forecasting Data
    df_arima = pd.read_csv("arima_forecast.csv", parse_dates=["Date"], index_col="Date")
    df_lstm = pd.read_csv("lstm_forecast.csv", parse_dates=["Date"], index_col="Date")
    df_prophet = pd.read_csv("prophet_forecast.csv", parse_dates=["Date"], index_col="Date")

    # Load Sentiment Data
    df_sentiment = pd.read_csv("crypto_sentiment.csv")

    # Fetch Current Bitcoin Price (this will be cached for 5 minutes)
    current_bitcoin_price = get_current_bitcoin_price()

    # ---- Streamlit UI ----
    st.title("📈 Cryptocurrency Price Forecasting & Sentiment Analysis")
    st.write("Analyze Bitcoin trends using ARIMA, LSTM, Prophet, and sentiment analysis from Twitter.")

    # ---- Current Bitcoin Price ----
    st.subheader("Current Bitcoin Price (USD)")

    # Display live Bitcoin price in default color, center-aligned
    st.markdown(f"<h2 style='text-align: center; font-weight: bold;'>${current_bitcoin_price:,.2f}</h2>", unsafe_allow_html=True)

    # ---- Bitcoin Price Data ----
    st.subheader("Bitcoin Price Data")
    st.write("Here is the raw Bitcoin price data used for analysis:")
    st.dataframe(df_prices.tail())  # Show last few rows of price data

    # ---- Bitcoin Price Trend ----
    st.subheader("Bitcoin Price Trend")
    st.line_chart(df_prices["Price"])

    # ---- ARIMA Forecast ----
    st.subheader("ARIMA Model Prediction")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index, df_prices["Price"], label="Actual Price", color="blue")
    ax.plot(df_arima.index, df_arima["Forecast"], label="ARIMA Forecast", linestyle="dashed", color="red")
    ax.legend()
    st.pyplot(fig)

    # ---- LSTM Forecast ----
    st.subheader("LSTM Model Prediction")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index, df_prices["Price"], label="Actual Price", color="blue")
    ax.plot(df_lstm.index, df_lstm["Forecast"], label="LSTM Forecast", linestyle="dashed", color="green")
    ax.legend()
    st.pyplot(fig)

    # ---- Prophet Forecast ----
    st.subheader("Prophet Model Prediction")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index, df_prices["Price"], label="Actual Price", color="blue")
    ax.plot(df_prophet.index, df_prophet["Forecast"], label="Prophet Forecast", linestyle="dashed", color="purple")
    ax.legend()
    st.pyplot(fig)

    # ---- Sentiment Analysis ----
    st.subheader("Crypto Market Sentiment Analysis")
