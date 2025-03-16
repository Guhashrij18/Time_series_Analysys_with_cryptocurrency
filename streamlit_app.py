import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import joblib
from sklearn.preprocessing import MinMaxScaler

# 🚀 Set Streamlit Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Live Crypto Predictions", layout="wide")

# ---- Cache the Live Bitcoin Price ----
@st.cache_data(ttl=60*5)  # Cache for 5 minutes (300 seconds)
def get_current_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data["bitcoin"]["usd"]  # Extract the current price

# ---- Load Data ----
@st.cache_data
def load_data():
    try:
        df_prices = pd.read_csv("bitcoin_prices.csv", parse_dates=["Date"], index_col="Date")
        df_sentiment = pd.read_csv("crypto_sentiment.csv")

        # Load Forecast Data (Corrected)
        df_arima = pd.read_csv("arima_forecast_corrected.csv", parse_dates=["Date"], index_col="Date")
        df_lstm = pd.read_csv("lstm_forecast_corrected.csv", parse_dates=["Date"], index_col="Date")
        df_prophet = pd.read_csv("prophet_forecast_corrected.csv", parse_dates=["Date"], index_col="Date")

        # ✅ Load MinMaxScaler for Inverse Transformation
        scaler = joblib.load("scaler.pkl")
        df_lstm["Forecast"] = scaler.inverse_transform(df_lstm[["Forecast"]])
        df_prophet["Forecast"] = scaler.inverse_transform(df_prophet[["Forecast"]])

        return df_prices, df_arima, df_lstm, df_prophet, df_sentiment
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None, None, None

# Load Data
df_prices, df_arima, df_lstm, df_prophet, df_sentiment = load_data()
current_bitcoin_price = get_current_bitcoin_price()

# ---- Streamlit UI ----
st.title("📊 Cryptocurrency Live Forecast & Sentiment Analysis")

# ---- Display Live Bitcoin Price ----
st.subheader("🔴 Live Bitcoin Price (USD)")
st.metric(label="Current Bitcoin Price (USD)", value=f"${current_bitcoin_price:,.2f}")

# ---- Bitcoin Price Trend ----
if df_prices is not None:
    st.subheader("📈 Bitcoin Price Trend (All Data)")
    st.line_chart(df_prices["Price"])

# ---- ARIMA Model Forecast ----
if df_arima is not None:
    st.subheader("🔮 ARIMA Model Live Prediction")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index, df_prices["Price"], label="Actual Price", color="blue")
    ax.plot(df_arima.index, df_arima["Forecast"], label="ARIMA Forecast", linestyle="dashed", color="red")
    ax.legend()
    st.pyplot(fig)

# ---- LSTM Model Forecast ----
if df_lstm is not None:
    st.subheader("🤖 LSTM Model Live Prediction")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index, df_prices["Price"], label="Actual Price", color="blue")
    ax.plot(df_lstm.index, df_lstm["Forecast"], label="LSTM Forecast", linestyle="dashed", color="green")
    ax.legend()
    st.pyplot(fig)

# ---- Prophet Model Forecast ----
if df_prophet is not None:
    st.subheader("📊 Prophet Model Live Prediction")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index, df_prices["Price"], label="Actual Price", color="blue")
    ax.plot(df_prophet.index, df_prophet["Forecast"], label="Prophet Forecast", linestyle="dashed", color="purple")
    ax.legend()
    st.pyplot(fig)

# ---- Live Sentiment Analysis ----
if df_sentiment is not None:
    st.subheader("📢 Live Sentiment Analysis of Bitcoin Tweets")
    st.write("Tracking the latest sentiment analysis based on Bitcoin-related tweets.")

    # Sentiment Data Summary
    positive_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] > 0])
    neutral_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] == 0])
    negative_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] < 0])

    # Sentiment Bar Chart
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    ax.bar(["Positive", "Neutral", "Negative"], [positive_tweets, neutral_tweets, negative_tweets], color=["green", "gray", "red"])
    ax.set_ylabel("Number of Tweets")
    ax.set_title("Sentiment Analysis of Bitcoin Tweets")
    st.pyplot(fig)

    # Overall Sentiment Score
    avg_sentiment = df_sentiment["Sentiment Score"].mean()
    st.subheader("📰 Overall Market Sentiment")
    if avg_sentiment > 0:
        st.success(f"🟢 *Positive Market Sentiment* (Score: {avg_sentiment:.2f})")
    elif avg_sentiment < 0:
        st.error(f"🔴 *Negative Market Sentiment* (Score: {avg_sentiment:.2f})")
    else:
        st.warning(f"⚪ *Neutral Market Sentiment* (Score: {avg_sentiment:.2f})")

# ---- Footer ----
st.markdown("🚀 Developed by Your Name | Powered by Streamlit, Plotly & Machine Learning")
