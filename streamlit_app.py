import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# ---- Load Data ----
def load_data(filename):
    """Loads a CSV file into a DataFrame with error handling."""
    try:
        df = pd.read_csv(filename, parse_dates=["Date"], index_col="Date")
        return df
    except FileNotFoundError:
        st.error(f"❌ `{filename}` not found! Please check your files.")
        return None

# Load all datasets
df_prices = load_data("bitcoin_prices.csv")
df_arima = load_data("arima_forecast.csv")
df_lstm = load_data("lstm_forecast.csv")
df_prophet = load_data("prophet_forecast.csv")
df_sentiment = pd.read_csv("crypto_sentiment.csv")  # Sentiment data (no date column)

# ---- Streamlit UI ----
st.title("📈 Cryptocurrency Price Forecasting & Sentiment Analysis")

# ---- Live Bitcoin Price ----
st.subheader("💰 Live Bitcoin Price (USD)")

def get_live_price():
    """Fetches the current Bitcoin price from CoinGecko API."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin", "vs_currencies": "usd"}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()["bitcoin"]["usd"]
    except requests.RequestException:
        return None

live_price = get_live_price()
if live_price:
    st.metric(label="Current Bitcoin Price (USD)", value=f"${live_price}")
else:
    st.error("⚠️ Failed to fetch live price. Try again later.")

# ---- Bitcoin Price Data ----
if df_prices is not None:
    st.subheader("📊 Bitcoin Price Data (Last 100 Days)")
    st.dataframe(df_prices.tail(100))

    # ---- Bitcoin Price Trend ----
    st.subheader("📈 Bitcoin Price Trend")
    st.line_chart(df_prices["Price"])

    # ---- Forecasting Models ----
    def plot_forecast(actual_df, forecast_df, title, color):
        """Plot actual vs. forecasted prices."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(actual_df.index[-100:], actual_df["Price"].iloc[-100:], label="Actual Price", color="blue")
        ax.plot(forecast_df.index, forecast_df["Forecast"], label=f"{title} Forecast", linestyle="dashed", color=color)
        ax.legend()
        st.pyplot(fig)

    # ---- ARIMA Forecast ----
    if df_arima is not None:
        st.subheader("🔮 ARIMA Model Prediction")
        plot_forecast(df_prices, df_arima, "ARIMA", "red")

    # ---- LSTM Forecast ----
    if df_lstm is not None:
        st.subheader("🤖 LSTM Model Prediction")
        plot_forecast(df_prices, df_lstm, "LSTM", "green")

    # ---- Prophet Forecast ----
    if df_prophet is not None:
        st.subheader("📊 Prophet Model Prediction")
        plot_forecast(df_prices, df_prophet, "Prophet", "purple")

# ---- Sentiment Analysis ----
if df_sentiment is not None:
    st.subheader("📢 Crypto Market Sentiment Analysis")

    # Display Sentiment Data
    st.write(df_sentiment.head())

    # Calculate Sentiment Distribution
    positive_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] > 0])
    neutral_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] == 0])
    negative_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] < 0])

    # Show Sentiment Distribution Chart
    st.subheader("📊 Sentiment Distribution")
    fig, ax = plt.subplots()
    ax.bar(["Positive", "Neutral", "Negative"], [positive_tweets, neutral_tweets, negative_tweets], color=["green", "gray", "red"])
    ax.set_ylabel("Number of Tweets")
    ax.set_title("Sentiment Analysis of Bitcoin Tweets")
    st.pyplot(fig)

    # Show Overall Market Sentiment
    avg_sentiment = df_sentiment["Sentiment Score"].mean()
    st.subheader("📢 Overall Crypto Market Sentiment")
    if avg_sentiment > 0:
        st.success(f"🟢 **Positive Market Sentiment** (Score: {avg_sentiment:.2f})")
    elif avg_sentiment < 0:
        st.error(f"🔴 **Negative Market Sentiment** (Score: {avg_sentiment:.2f})")
    else:
        st.info(f"⚪ **Neutral Market Sentiment** (Score: {avg_sentiment:.2f})")

    # Dropdown to Filter Tweets by Sentiment
    sentiment_filter = st.selectbox("🔍 Select Sentiment to View Tweets", ["All", "Positive", "Neutral", "Negative"])
    filtered_df = df_sentiment[df_sentiment["Sentiment Score"] > 0] if sentiment_filter == "Positive" else \
                  df_sentiment[df_sentiment["Sentiment Score"] < 0] if sentiment_filter == "Negative" else \
                  df_sentiment[df_sentiment["Sentiment Score"] == 0] if sentiment_filter == "Neutral" else df_sentiment

    # Display Filtered Tweets
    st.subheader(f"📢 {sentiment_filter} Tweets")
    st.write(filtered_df[["Tweet", "Sentiment Score"]])
