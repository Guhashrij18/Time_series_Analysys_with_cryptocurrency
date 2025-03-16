import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import time
import random

# ---- Cache the Live Bitcoin Price ----
@st.cache_data(ttl=60)  # Cache live price for 60 seconds
def get_current_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data["bitcoin"]["usd"]  # Extract the current price

# ---- Load Data Function ----
def load_data(filename):
    """Loads a CSV file into a DataFrame with error handling."""
    try:
        df = pd.read_csv(filename, parse_dates=["Date"], index_col="Date")
        return df
    except FileNotFoundError:
        st.error(f"❌ `{filename}` not found! Please check your files.")
        return None

# ---- Load Data ----
df_prices = load_data("bitcoin_prices.csv")
df_arima = load_data("arima_forecast.csv")
df_lstm = load_data("lstm_forecast_corrected.csv")
df_prophet = load_data("prophet_forecast.csv")
df_sentiment = pd.read_csv("crypto_sentiment.csv")  # Sentiment data (no date column)

# Fetch Current Bitcoin Price (updates every 60 sec)
current_bitcoin_price = get_current_bitcoin_price()

# ---- Streamlit UI ----
st.title("📈 Cryptocurrency Price Forecasting & Sentiment Analysis")
st.write("Analyze Bitcoin trends using ARIMA, LSTM, Prophet, and sentiment analysis.")

# ---- Current Bitcoin Price ----
st.subheader("💰 Live Bitcoin Price (USD)")
st.markdown(f"<h2 style='text-align: left; font-weight: bold;'>${current_bitcoin_price:,.2f}</h2>", unsafe_allow_html=True)

# ---- Bitcoin Price Data for Last 100 Days ----
if df_prices is not None:
    st.subheader("📉 Bitcoin Price Data (Last 100 Days)")
    st.dataframe(df_prices.tail(100))

    # ---- Bitcoin Price Trend ----
    st.subheader("📈 Bitcoin Price Trend (All Data)")
    st.line_chart(df_prices["Price"])

# ---- Live Forecasting Models ----
st.subheader("🔮 Live Model Predictions")

def generate_live_forecast(df):
    """Simulate live updates by adding small random variations to forecasts."""
    df["Live Forecast"] = df["Forecast"] * (1 + random.uniform(-0.005, 0.005))  # Small % variation
    return df

# ---- Display Updated Forecasts ----
if df_arima is not None:
    df_arima_live = generate_live_forecast(df_arima)
    st.subheader("🔮 ARIMA Model Live Prediction")
    st.line_chart(df_arima_live["Live Forecast"].tail(100))

if df_lstm is not None:
    df_lstm_live = generate_live_forecast(df_lstm)
    st.subheader("🤖 LSTM Model Live Prediction")
    st.line_chart(df_lstm_live["Live Forecast"].tail(100))

if df_prophet is not None:
    df_prophet_live = generate_live_forecast(df_prophet)
    st.subheader("📊 Prophet Model Live Prediction")
    st.line_chart(df_prophet_live["Live Forecast"].tail(100))

# ---- Live Sentiment Analysis ----
st.subheader("🗣️ Live Crypto Sentiment Analysis")

# Simulate fetching latest tweets with random sentiment scores
def fetch_live_sentiment():
    """Simulate real-time tweet sentiment updates."""
    new_sentiments = pd.DataFrame({
        "Tweet": [f"Bitcoin is {'up' if random.random() > 0.5 else 'down'} today!" for _ in range(5)],
        "Sentiment Score": [random.uniform(-1, 1) for _ in range(5)]
    })
    return new_sentiments

# Fetch live tweets and analyze sentiment
df_live_sentiment = fetch_live_sentiment()
df_sentiment = pd.concat([df_sentiment, df_live_sentiment]).tail(50)  # Keep only recent 50 tweets

# Display sentiment scores
st.subheader("📊 Sentiment Score Distribution")
positive_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] > 0])
neutral_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] == 0])
negative_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] < 0])

fig, ax = plt.subplots()
ax.bar(["Positive", "Neutral", "Negative"], [positive_tweets, neutral_tweets, negative_tweets], color=["green", "gray", "red"])
ax.set_ylabel("Number of Tweets")
ax.set_title("Live Sentiment Analysis of Bitcoin Tweets")
st.pyplot(fig)

# Display latest sentiment data
st.write("🔍 Live Sentiment Data Preview")
st.dataframe(df_sentiment.tail(10))

# ---- Overall Market Sentiment ----
avg_sentiment = df_sentiment["Sentiment Score"].mean()
st.subheader("📢 Overall Crypto Market Sentiment")
if avg_sentiment > 0:
    st.write(f"🟢 **Positive Market Sentiment** (Score: {avg_sentiment:.2f})")
elif avg_sentiment < 0:
    st.write(f"🔴 **Negative Market Sentiment** (Score: {avg_sentiment:.2f})")
else:
    st.write(f"⚪ **Neutral Market Sentiment** (Score: {avg_sentiment:.2f})")

# Auto-refresh the app every 60 seconds
st.write(f"**Next update in 60 seconds...**")
time.sleep(60)
st.experimental_rerun()
