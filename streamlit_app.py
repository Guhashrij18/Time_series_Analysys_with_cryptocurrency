import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import joblib
from sklearn.preprocessing import MinMaxScaler

# ---- Cache the Live Bitcoin Price ----
@st.cache_data(ttl=60*5)  # Cache for 5 minutes (300 seconds)
def get_current_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return data["bitcoin"]["usd"]  # Extract the current price

# ---- Load Data & Models ----
@st.cache_data
def load_data():
    try:
        df_prices = pd.read_csv("bitcoin_prices.csv", parse_dates=["Date"], index_col="Date")
        df_sentiment = pd.read_csv("crypto_sentiment.csv")

        # Load Forecast Data
        df_arima = pd.read_csv("arima_forecast.csv", parse_dates=["Date"], index_col="Date")
        df_lstm = pd.read_csv("lstm_forecast.csv", parse_dates=["Date"], index_col="Date")
        df_prophet = pd.read_csv("prophet_forecast.csv", parse_dates=["Date"], index_col="Date")

        # Load the MinMaxScaler (for inverse transforming LSTM & Prophet)
        scaler = joblib.load("scaler.pkl")  # Ensure you have saved the same scaler used during training

        # Inverse transform LSTM & Prophet predictions
        df_lstm["Forecast"] = scaler.inverse_transform(df_lstm[["Forecast"]])
        df_prophet["Forecast"] = scaler.inverse_transform(df_prophet[["Forecast"]])

        # Save corrected forecasts
        df_arima.to_csv("arima_forecast_corrected.csv")
        df_lstm.to_csv("lstm_forecast_corrected.csv")
        df_prophet.to_csv("prophet_forecast_corrected.csv")

        return df_prices, df_arima, df_lstm, df_prophet, df_sentiment
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

df_prices, df_arima, df_lstm, df_prophet, df_sentiment = load_data()
current_bitcoin_price = get_current_bitcoin_price()

# ---- Streamlit UI ----
st.set_page_config(page_title="Live Crypto Predictions", layout="wide")
st.title("📊 Cryptocurrency Live Forecast & Sentiment Analysis")

# ---- Display Live Bitcoin Price ----
st.subheader("🔴 Live Bitcoin Price (USD)")
st.markdown(f"<h2 style='font-weight: bold;'>${current_bitcoin_price:,.2f}</h2>", unsafe_allow_html=True)

# ---- Bitcoin Price Trend ----
st.subheader("📈 Bitcoin Price Trend (All Data)")
st.line_chart(df_prices["Price"])

# ---- ARIMA Model Forecast ----
st.subheader("🔮 ARIMA Model Live Prediction")
fig_arima = px.line(df_arima, x=df_arima.index, y="Forecast", title="ARIMA Forecast", markers=True)
fig_arima.add_scatter(x=df_prices.index, y=df_prices["Price"], mode="lines", name="Actual Price", line=dict(color="blue"))
fig_arima.update_traces(line=dict(color="red"))
st.plotly_chart(fig_arima, use_container_width=True)

# ---- LSTM Model Forecast ----
st.subheader("🤖 LSTM Model Live Prediction")
fig_lstm = px.line(df_lstm, x=df_lstm.index, y="Forecast", title="LSTM Forecast", markers=True)
fig_lstm.add_scatter(x=df_prices.index, y=df_prices["Price"], mode="lines", name="Actual Price", line=dict(color="blue"))
fig_lstm.update_traces(line=dict(color="green"))
st.plotly_chart(fig_lstm, use_container_width=True)

# ---- Prophet Model Forecast ----
st.subheader("📊 Prophet Model Live Prediction")
fig_prophet = px.line(df_prophet, x=df_prophet.index, y="Forecast", title="Prophet Forecast", markers=True)
fig_prophet.add_scatter(x=df_prices.index, y=df_prices["Price"], mode="lines", name="Actual Price", line=dict(color="blue"))
fig_prophet.update_traces(line=dict(color="purple"))
st.plotly_chart(fig_prophet, use_container_width=True)

# ---- Live Sentiment Analysis ----
st.subheader("📢 Live Sentiment Analysis of Bitcoin Tweets")
st.write("Tracking the latest sentiment analysis based on Bitcoin-related tweets.")

# Sentiment Data Summary
positive_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] > 0])
neutral_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] == 0])
negative_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] < 0])

# Sentiment Bar Chart
fig_sentiment = px.bar(
    x=["Positive", "Neutral", "Negative"],
    y=[positive_tweets, neutral_tweets, negative_tweets],
    color=["green", "gray", "red"],
    title="Sentiment Distribution"
)
st.plotly_chart(fig_sentiment, use_container_width=True)

# Overall Sentiment Score
avg_sentiment = df_sentiment["Sentiment Score"].mean()
st.subheader("📰 Overall Market Sentiment")
if avg_sentiment > 0:
    st.success(f"🟢 *Positive Market Sentiment* (Score: {avg_sentiment:.2f})")
elif avg_sentiment < 0:
    st.error(f"🔴 *Negative Market Sentiment* (Score: {avg_sentiment:.2f})")
else:
    st.warning(f"⚪ *Neutral Market Sentiment* (Score: {avg_sentiment:.2f})")

# Show latest tweets
st.subheader("📌 Latest Tweets & Sentiment")
sentiment_filter = st.selectbox("Filter Tweets by Sentiment", ["All", "Positive", "Neutral", "Negative"])
if sentiment_filter == "Positive":
    filtered_df = df_sentiment[df_sentiment["Sentiment Score"] > 0]
elif sentiment_filter == "Negative":
    filtered_df = df_sentiment[df_sentiment["Sentiment Score"] < 0]
elif sentiment_filter == "Neutral":
    filtered_df = df_sentiment[df_sentiment["Sentiment Score"] == 0]
else:
    filtered_df = df_sentiment

st.write(filtered_df[["Tweet", "Sentiment Score"]].head(10))  # Show 10 recent tweets

