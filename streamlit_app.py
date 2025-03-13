import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
    st.title("Cryptocurrency Price Forecasting & Sentiment Analysis")
    st.write("This dashboard shows Bitcoin price trends, forecasts, and sentiment analysis.")

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

except FileNotFoundError as e:
    st.error(f"Error loading data: {e}")

