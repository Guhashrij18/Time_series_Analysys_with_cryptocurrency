import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import openai

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

    # ---- Streamlit UI ----
    st.title("📈 Cryptocurrency Price Forecasting & Sentiment Analysis")
    st.write("This dashboard shows Bitcoin price trends, forecasts, and sentiment analysis.")

    # ---- Live Bitcoin Price ----
    st.subheader("💰 Live Bitcoin Price (USD)")

    def get_live_price():
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin", "vs_currencies": "usd"}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()["bitcoin"]["usd"]
        return None

    live_price = get_live_price()
    if live_price:
        st.metric(label="Current Bitcoin Price (USD)", value=f"${live_price}")
    else:
        st.error("⚠ Failed to fetch live price. Try again later.")

    # ---- Bitcoin Price Data Table ----
    st.subheader("📋 Bitcoin Price Data (Last 100 Days)")
    st.write("This table displays the last 100 days of Bitcoin price data.")
    st.dataframe(df_prices.tail(100))  # Show last 100 rows

    # ---- Bitcoin Price Trend ----
    st.subheader("📊 Bitcoin Price Trend")
    st.line_chart(df_prices["Price"])

    # ---- ARIMA Forecast ----
    st.subheader("🔮 ARIMA Model Prediction")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index[-100:], df_prices["Price"].iloc[-100:], label="Actual Price", color="blue")
    ax.plot(df_arima.index[-100:], df_arima["Forecast"].iloc[-100:], label="ARIMA Forecast", linestyle="dashed", color="red")
    ax.legend()
    st.pyplot(fig)

    # ---- LSTM Forecast ----
    st.subheader("🤖 LSTM Model Prediction")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index[-100:], df_prices["Price"].iloc[-100:], label="Actual Price", color="blue")
    ax.plot(df_lstm.index[-100:], df_lstm["Forecast"].iloc[-100:], label="LSTM Forecast", linestyle="dashed", color="green")
    ax.legend()
    st.pyplot(fig)

    # ---- Prophet Forecast ----
    st.subheader("🔥 Prophet Model Prediction")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index[-100:], df_prices["Price"].iloc[-100:], label="Actual Price", color="blue")
    ax.plot(df_prophet.index[-100:], df_prophet["Forecast"].iloc[-100:], label="Prophet Forecast", linestyle="dashed", color="purple")
    ax.legend()
    st.pyplot(fig)

    # ---- Sentiment Analysis ----
    st.subheader("📢 Crypto Market Sentiment Analysis")
    st.write("Sentiment analysis of Bitcoin-related tweets.")

    # Show Sentiment Data
    st.subheader("🔍 Sentiment Data Preview")
    st.write(df_sentiment.head())  # Show first few tweets & scores

    # Show All Tweets in a Table
    st.subheader("📜 All Collected Tweets")
    st.dataframe(df_sentiment)  # Show all tweets

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
        st.write(f"🟢 *Positive Market Sentiment* (Score: {avg_sentiment:.2f})")
    elif avg_sentiment < 0:
        st.write(f"🔴 *Negative Market Sentiment* (Score: {avg_sentiment:.2f})")
    else:
        st.write(f"⚪ *Neutral Market Sentiment* (Score: {avg_sentiment:.2f})")

    # Dropdown to Filter Tweets by Sentiment
    sentiment_filter = st.selectbox("🔍 Select Sentiment to View Tweets", ["All", "Positive", "Neutral", "Negative"])
    if sentiment_filter == "Positive":
        filtered_df = df_sentiment[df_sentiment["Sentiment Score"] > 0]
    elif sentiment_filter == "Negative":
        filtered_df = df_sentiment[df_sentiment["Sentiment Score"] < 0]
    elif sentiment_filter == "Neutral":
        filtered_df = df_sentiment[df_sentiment["Sentiment Score"] == 0]
    else:
        filtered_df = df_sentiment

    # Display Filtered Tweets
    st.subheader(f"📢 {sentiment_filter} Tweets")
    st.write(filtered_df[["Tweet", "Sentiment Score"]])
