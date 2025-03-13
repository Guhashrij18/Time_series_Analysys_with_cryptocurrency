import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load Data Files
price_file = "bitcoin_prices.csv"
prophet_forecast_file = "prophet_forecast.csv"
arima_forecast_file = "arima_forecast.csv"
lstm_forecast_file = "lstm_forecast.csv"
sentiment_file = "sentiment_analysis.csv"

st.title("📈 Cryptocurrency Analysis Dashboard")

### 📌 LIVE BITCOIN PRICE ###
try:
    df_prices = pd.read_csv(price_file, parse_dates=["Date"], index_col="Date")
    st.subheader("💰 Live Bitcoin Price (USD)")
    latest_price = df_prices["Price"].iloc[-1]
    st.metric(label="Current Bitcoin Price", value=f"${latest_price:,.2f}")

    st.subheader("📉 Bitcoin Price Trend (Last 100 Days)")
    st.line_chart(df_prices["Price"].tail(100))

except Exception as e:
    st.error(f"⚠️ Error loading Bitcoin price data: {e}")

### 📌 PROPHET FORECAST ###
try:
    df_prophet = pd.read_csv(prophet_forecast_file)
    df_prophet['Date'] = pd.to_datetime(df_prophet['Date'])

    st.subheader("📊 Bitcoin Price Forecast (Prophet)")
    st.line_chart(df_prophet.set_index("Date").tail(100))

    st.write("🔍 Prophet Forecast Data Preview")
    st.dataframe(df_prophet.head())

except Exception as e:
    st.error(f"⚠️ Error loading Prophet forecast data: {e}")

### 📌 ARIMA FORECAST ###
try:
    df_arima = pd.read_csv(arima_forecast_file)
    df_arima['Date'] = pd.to_datetime(df_arima['Date'])

    st.subheader("📊 Bitcoin Price Forecast (ARIMA)")
    st.line_chart(df_arima.set_index("Date").tail(100))

    st.write("🔍 ARIMA Forecast Data Preview")
    st.dataframe(df_arima.head())

except Exception as e:
    st.error(f"⚠️ Error loading ARIMA forecast data: {e}")

### 📌 LSTM FORECAST ###
try:
    df_lstm = pd.read_csv(lstm_forecast_file)
    df_lstm['Date'] = pd.to_datetime(df_lstm['Date'])

    st.subheader("📊 Bitcoin Price Forecast (LSTM)")
    st.line_chart(df_lstm.set_index("Date").tail(100))

    st.write("🔍 LSTM Forecast Data Preview")
    st.dataframe(df_lstm.head())

except Exception as e:
    st.error(f"⚠️ Error loading LSTM forecast data: {e}")

### 📌 SENTIMENT ANALYSIS ###
try:
    df_sentiment = pd.read_csv(sentiment_file)
    sentiment_counts = df_sentiment['Sentiment'].value_counts()

    st.subheader("🗣️ Bitcoin Sentiment Analysis")
    st.bar_chart(sentiment_counts)

    st.write("🔍 Sentiment Data Preview")
    st.dataframe(df_sentiment.head())

except Exception as e:
    st.error(f"⚠️ Error loading sentiment data: {e}")

st.write("This dashboard provides Bitcoin price trends, forecasts using Prophet, ARIMA, and LSTM, and sentiment analysis.")
