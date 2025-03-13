import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# ---- Streamlit UI ----
st.title("📈 Cryptocurrency Price Forecasting & Sentiment Analysis")

# ---- Fetch Live Bitcoin Price ----
st.subheader("💰 Live Bitcoin Price (USD)")

def get_live_price():
    """Fetches the current Bitcoin price from CoinGecko API."""
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": "bitcoin", "vs_currencies": "usd"}
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error for bad response
        return response.json()["bitcoin"]["usd"]
    except requests.RequestException:
        return None

live_price = get_live_price()
if live_price:
    st.metric(label="Current Bitcoin Price (USD)", value=f"${live_price}")
else:
    st.error("⚠️ Failed to fetch live price. Try again later.")

# ---- Load Bitcoin Price Data ----
st.subheader("📋 Bitcoin Price Data (Last 100 Days)")

try:
    df_prices = pd.read_csv("bitcoin_prices.csv", parse_dates=["Date"], index_col="Date")
    st.dataframe(df_prices.tail(100))  # Show last 100 rows

    # ---- Bitcoin Price Trend ----
    st.subheader("📊 Bitcoin Price Trend")
    st.line_chart(df_prices["Price"])

    # ---- Moving Averages ----
    st.subheader("📈 Bitcoin Price with Moving Averages")
    df_prices["SMA_50"] = df_prices["Price"].rolling(window=50).mean()
    df_prices["EMA_20"] = df_prices["Price"].ewm(span=20, adjust=False).mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index, df_prices["Price"], label="Bitcoin Price", color="blue")
    ax.plot(df_prices.index, df_prices["SMA_50"], label="50-day SMA", linestyle="dashed", color="red")
    ax.plot(df_prices.index, df_prices["EMA_20"], label="20-day EMA", linestyle="dashed", color="green")
    ax.legend()
    st.pyplot(fig)

    # ---- Load Forecasting Data ----
    df_arima = pd.read_csv("arima_forecast.csv", parse_dates=["Date"], index_col="Date")
    df_lstm = pd.read_csv("lstm_forecast.csv", parse_dates=["Date"], index_col="Date")
    df_prophet = pd.read_csv("prophet_forecast.csv")

    # ---- Fix Prophet Model Data ----
    if "Date" in df_prophet.columns and "Forecast" in df_prophet.columns:
        df_prophet["Date"] = pd.to_datetime(df_prophet["Date"])  # Convert Date column to DateTime
        df_prophet.set_index("Date", inplace=True)  # Set Date as index
    else:
        st.error("⚠️ Prophet forecast data does not contain 'Date' and 'Forecast' columns!")

    # ---- Forecasting Period Selection ----
    st.subheader("⏳ Choose Forecasting Period")
    forecast_days = st.slider("Select number of days to forecast", min_value=30, max_value=180, step=30, value=60)

    # ---- ARIMA Forecast ----
    st.subheader(f"🔮 ARIMA Model Prediction for {forecast_days} Days")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index[-100:], df_prices["Price"].iloc[-100:], label="Actual Price", color="blue")
    ax.plot(df_arima.index[:forecast_days], df_arima["Forecast"].iloc[:forecast_days], label="ARIMA Forecast", linestyle="dashed", color="red")
    ax.legend()
    st.pyplot(fig)

    # ---- LSTM Forecast ----
    st.subheader(f"🤖 LSTM Model Prediction for {forecast_days} Days")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index[-100:], df_prices["Price"].iloc[-100:], label="Actual Price", color="blue")
    ax.plot(df_lstm.index[:forecast_days], df_lstm["Forecast"].iloc[:forecast_days], label="LSTM Forecast", linestyle="dashed", color="green")
    ax.legend()
    st.pyplot(fig)

    # ---- Prophet Forecast (Fixed) ----
    st.subheader(f"🔥 Prophet Model Prediction for {forecast_days} Days")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index[-100:], df_prices["Price"].iloc[-100:], label="Actual Price", color="blue")

    if not df_prophet.empty and "Forecast" in df_prophet.columns:
        ax.plot(df_prophet.index[:forecast_days], df_prophet["Forecast"].iloc[:forecast_days], label="Prophet Forecast", linestyle="dashed", color="purple")
    else:
        st.error("⚠️ Prophet forecast data is empty or missing 'Forecast' column!")

    ax.legend()
    st.pyplot(fig)

    # ---- Load Sentiment Data ----
    df_sentiment = pd.read_csv("crypto_sentiment.csv")

    # ---- Sentiment Analysis ----
    st.subheader("📢 Crypto Market Sentiment Analysis")

    # Show Sentiment Data
    st.subheader("🔍 Sentiment Data Preview")
    st.write(df_sentiment.head())

    # Dropdown to Filter Tweets
    sentiment_filter = st.selectbox("🔍 Select Sentiment to View Tweets", ["All", "Positive", "Neutral", "Negative"])
    filtered_df = df_sentiment[df_sentiment["Sentiment Score"] > 0] if sentiment_filter == "Positive" else \
                  df_sentiment[df_sentiment["Sentiment Score"] < 0] if sentiment_filter == "Negative" else \
                  df_sentiment[df_sentiment["Sentiment Score"] == 0] if sentiment_filter == "Neutral" else df_sentiment
    st.write(filtered_df[["Tweet", "Sentiment Score"]])

except FileNotFoundError as e:
    st.error(f"❌ Error loading data: {e}")
