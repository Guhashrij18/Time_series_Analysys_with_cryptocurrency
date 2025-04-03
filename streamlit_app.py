import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# ---- Set Page Configuration ----
st.set_page_config(page_title="Bitcoin Forecast & Sentiment Analysis", layout="wide")

# ---- Cache the Live Bitcoin Price ----
@st.cache_data(ttl=300)  # Cache for 5 minutes (300 seconds)
def get_current_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad responses (4xx, 5xx)
        data = response.json()
        return data.get("bitcoin", {}).get("usd", "N/A")
    
    except requests.exceptions.RequestException:
        st.error("⚠️ Error fetching Bitcoin price. Please try again later.")
        return "N/A"

# ---- Load Data with Error Handling ----
def load_csv(filename, parse_dates=["Date"]):
    try:
        df = pd.read_csv(filename, parse_dates=parse_dates, index_col="Date")
        return df
    except FileNotFoundError:
        st.error(f"⚠️ Error: `{filename}` not found! Please check the file path.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading `{filename}`: {str(e)}")
        return None

# Load Bitcoin Price Data
df_prices = load_csv("bitcoin_prices.csv")

# Load Forecasting Data
df_arima = load_csv("arima_forecast.csv")
df_lstm = load_csv("lstm_forecast.csv")
df_prophet = load_csv("prophet_forecast.csv")

# Load Sentiment Data
try:
    df_sentiment = pd.read_csv("crypto_sentiment.csv")
    
    # Ensure required columns exist
    if not {"Date", "Avg Sentiment Score", "Tweet"}.issubset(df_sentiment.columns):
        st.error("⚠️ Sentiment data is missing required columns. Showing default empty data.")
        df_sentiment = pd.DataFrame(columns=["Date", "Avg Sentiment Score", "Tweet"])
    
    # Fill NaN values
    df_sentiment["Avg Sentiment Score"].fillna(0, inplace=True)

except FileNotFoundError:
    st.error("⚠️ Error: `crypto_sentiment.csv` not found!")
    df_sentiment = None

# Fetch Current Bitcoin Price
current_bitcoin_price = get_current_bitcoin_price()

# ---- Streamlit UI ----
st.title("📈 Cryptocurrency Live Forecast & Sentiment Analysis")
st.write("Analyze Bitcoin trends using ARIMA, LSTM, Prophet, and sentiment analysis.")

# ---- Current Bitcoin Price ----
st.subheader("💰 Live Bitcoin Price (USD)")
if current_bitcoin_price == "N/A":
    st.warning("⚠️ Bitcoin price could not be retrieved. Please try again later.")
else:
    st.markdown(f"<h2 style='text-align: left; font-weight: bold;'>${current_bitcoin_price:,.2f}</h2>", unsafe_allow_html=True)

# ---- Bitcoin Price Data ----
if df_prices is not None:
    st.subheader("📊 Bitcoin Price Data (Last 100 Days)")
    st.dataframe(df_prices.tail(100))

    # ---- Bitcoin Price Trend ----
    st.subheader("📈 Bitcoin Price Trend (All Data)")
    st.line_chart(df_prices["Price"])

# ---- Forecasting Models ----
def plot_forecast(actual_df, forecast_df, title, color):
    """Helper function to plot forecast models."""
    if actual_df is not None and forecast_df is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(actual_df.index, actual_df["Price"], label="Actual Price", color="blue")
        ax.plot(forecast_df.index, forecast_df["Forecast"], label=title, linestyle="dashed", color=color)
        ax.legend()
        st.pyplot(fig)

st.subheader("🔮 Forecasting Models")
plot_forecast(df_prices, df_arima, "ARIMA Forecast", "red")
plot_forecast(df_prices, df_lstm, "LSTM Forecast", "green")
plot_forecast(df_prices, df_prophet, "Prophet Forecast", "purple")

# ---- Sentiment Analysis ----
if df_sentiment is not None and not df_sentiment.empty:
    st.subheader("🧐 Crypto Market Sentiment Analysis")

    # Show Sentiment Data
    st.subheader("📋 Sentiment Data Preview")
    st.dataframe(df_sentiment.tail(10))  # Show last few tweets & scores

    # Sentiment Distribution
    positive_tweets = len(df_sentiment[df_sentiment["Avg Sentiment Score"] > 0])
    neutral_tweets = len(df_sentiment[df_sentiment["Avg Sentiment Score"] == 0])
    negative_tweets = len(df_sentiment[df_sentiment["Avg Sentiment Score"] < 0])

    st.subheader("📊 Sentiment Distribution")
    fig, ax = plt.subplots()
    ax.bar(["Positive", "Neutral", "Negative"], [positive_tweets, neutral_tweets, negative_tweets], color=["green", "gray", "red"])
    ax.set_ylabel("Number of Tweets")
    ax.set_title("Sentiment Analysis of Bitcoin Tweets")
    st.pyplot(fig)

    # Show Overall Market Sentiment
    avg_sentiment = df_sentiment["Avg Sentiment Score"].mean()
    st.subheader("🧐 Overall Crypto Market Sentiment")
    if avg_sentiment > 0:
        st.success(f"🟢 **Positive Market Sentiment** (Score: {avg_sentiment:.2f})")
    elif avg_sentiment < 0:
        st.error(f"🔴 **Negative Market Sentiment** (Score: {avg_sentiment:.2f})")
    else:
        st.info(f"⚪ **Neutral Market Sentiment** (Score: {avg_sentiment:.2f})")

    # Dropdown to Filter Tweets by Sentiment
    sentiment_filter = st.selectbox("🔍 Select Sentiment to View Tweets", ["All", "Positive", "Neutral", "Negative"])

    if sentiment_filter == "Positive":
        filtered_df = df_sentiment[df_sentiment["Avg Sentiment Score"] > 0]
    elif sentiment_filter == "Negative":
        filtered_df = df_sentiment[df_sentiment["Avg Sentiment Score"] < 0]
    elif sentiment_filter == "Neutral":
        filtered_df = df_sentiment[df_sentiment["Avg Sentiment Score"] == 0]
    else:
        filtered_df = df_sentiment

    # Display Filtered Tweets
    st.subheader(f"📝 {sentiment_filter} Tweets")
    if not filtered_df.empty:
        st.dataframe(filtered_df[["Date", "Tweet", "Avg Sentiment Score"]])
    else:
        st.warning("⚠️ No tweets available for the selected sentiment.")

else:
    st.warning("⚠️ Sentiment data not available!")

