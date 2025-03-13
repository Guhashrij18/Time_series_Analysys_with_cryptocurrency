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
    st.markdown(f"<h2 style='text-align: left; font-weight: bold;'>${current_bitcoin_price:,.2f}</h2>", unsafe_allow_html=True)

    # ---- Bitcoin Price Data for Last 100 Days ----
    st.subheader("Bitcoin Price Data (Last 100 Days)")
    
    # Filter the last 100 rows of Bitcoin price data
    df_last_100_days = df_prices.tail(100)

    # Show last 100 days Bitcoin price data as a table
    st.dataframe(df_last_100_days)

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
    st.write("Sentiment analysis of Bitcoin-related tweets.")

    # Show Sentiment Data
    st.subheader("Sentiment Data Preview")
    st.write(df_sentiment.head())  # Show first few tweets & scores

    # Calculate Sentiment Distribution
    positive_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] > 0])
    neutral_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] == 0])
    negative_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] < 0])

    # Show Sentiment Distribution Chart
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    ax.bar(["Positive", "Neutral", "Negative"], [positive_tweets, neutral_tweets, negative_tweets], color=["green", "gray", "red"])
    ax.set_ylabel("Number of Tweets")
    ax.set_title("Sentiment Analysis of Bitcoin Tweets")
    st.pyplot(fig)

    # Show Overall Market Sentiment
    avg_sentiment = df_sentiment["Sentiment Score"].mean()
    st.subheader("Overall Crypto Market Sentiment")
    if avg_sentiment > 0:
        st.write(f"🟢 **Positive Market Sentiment** (Score: {avg_sentiment:.2f})")
    elif avg_sentiment < 0:
        st.write(f"🔴 **Negative Market Sentiment** (Score: {avg_sentiment:.2f})")
    else:
        st.write(f"⚪ **Neutral Market Sentiment** (Score: {avg_sentiment:.2f})")

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
    st.subheader(f"{sentiment_filter} Tweets")
    st.write(filtered_df[["Tweet", "Sentiment Score"]])

except FileNotFoundError as e:
    st.error(f"Error loading data: {e}")
