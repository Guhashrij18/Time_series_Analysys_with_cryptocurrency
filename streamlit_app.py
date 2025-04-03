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
        response = requests.get(url, timeout=10)  # Set timeout to avoid long waits
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        return data.get("bitcoin", {}).get("usd", None)  # Return None if price is missing
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            st.warning("⚠️ Too Many Requests. Please try again later.")
        else:
            st.error(f"⚠️ API Error: {e}")
        return None
    except requests.RequestException as e:
        st.error(f"⚠️ Error fetching Bitcoin price: {e}")
        return None

# ---- Load Data with Error Handling ----
def load_csv(filename, parse_dates=["Date"]):
    try:
        df = pd.read_csv(filename, parse_dates=parse_dates, index_col="Date")
        return df
    except FileNotFoundError:
        st.error(f"❌ File Not Found: `{filename}`")
        return None
    except pd.errors.ParserError:
        st.error(f"⚠️ Error parsing `{filename}`. Check CSV format!")
        return None
    except Exception as e:
        st.error(f"⚠️ Unexpected error loading `{filename}`: {str(e)}")
        return None

# Load Bitcoin Price Data
df_prices = load_csv("bitcoin_prices.csv")

# Load Forecasting Data
df_arima = load_csv("arima_forecast.csv")
df_lstm = load_csv("lstm_forecast.csv")
df_prophet = load_csv("prophet_forecast.csv")

# Load Sentiment Data (No Date Parsing)
df_sentiment = load_csv("crypto_sentiment.csv", parse_dates=None)

# Ensure Sentiment DataFrame has required columns
if df_sentiment is not None:
    required_columns = {"Date", "Tweet", "Avg Sentiment Score"}
    if not required_columns.issubset(df_sentiment.columns):
        st.warning("⚠️ Sentiment data is missing required columns. Showing default empty data.")
        df_sentiment = pd.DataFrame(columns=list(required_columns))  # Default empty DataFrame
    else:
        df_sentiment.rename(columns={"Avg Sentiment Score"}, inplace=True)
        df_sentiment["Avg Sentiment Score"].fillna(0, inplace=True)  # Replace NaN with 0
else:
    df_sentiment = pd.DataFrame(columns=["Date", "Tweet", "Avg Sentiment Score"])

# Fetch Current Bitcoin Price
current_bitcoin_price = get_current_bitcoin_price()

# ---- Streamlit UI ----
st.title("📈 Cryptocurrency Live Forecast & Sentiment Analysis")
st.write("Analyze Bitcoin trends using ARIMA, LSTM, Prophet, and sentiment analysis.")

# ---- Current Bitcoin Price ----
st.subheader("💰 Live Bitcoin Price (USD)")
if current_bitcoin_price is not None:
    st.markdown(f"<h2 style='text-align: left; font-weight: bold;'>${current_bitcoin_price:,.2f}</h2>", unsafe_allow_html=True)
else:
    st.warning("⚠️ Bitcoin price could not be retrieved. Please try again later.")

# ---- Bitcoin Price Data (Last 100 Days) ----
if df_prices is not None:
    st.subheader("📊 Bitcoin Price Data (Last 100 Days)")
    st.dataframe(df_prices.tail(100))

    # ---- Bitcoin Price Trend ----
    st.subheader("📉 Bitcoin Price Trend (All Data)")
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
    else:
        st.warning(f"⚠️ Unable to display {title} forecast due to missing data.")

st.subheader("🔮 Forecasting Models")
plot_forecast(df_prices, df_arima, "ARIMA Forecast", "red")
plot_forecast(df_prices, df_lstm, "LSTM Forecast", "green")
plot_forecast(df_prices, df_prophet, "Prophet Forecast", "purple")

# ---- Sentiment Analysis ----
if not df_sentiment.empty:
    st.subheader("🧐 Crypto Market Sentiment Analysis")

    # Show Sentiment Data
    st.subheader("📋 Sentiment Data Preview")
    st.write(df_sentiment.tail(10))  # Show last few tweets & scores

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
    st.subheader("💡 Overall Crypto Market Sentiment")
    if avg_sentiment > 0:
        st.success(f"🟢 **Positive Market Sentiment** (Score: {avg_sentiment:.2f})")
    elif avg_sentiment < 0:
        st.error(f"🔴 **Negative Market Sentiment** (Score: {avg_sentiment:.2f})")
    else:
        st.info(f"⚪ **Neutral Market Sentiment** (Score: {avg_sentiment:.2f})")

    # Dropdown to Filter Tweets by Sentiment
    sentiment_filter = st.selectbox("📌 Select Sentiment to View Tweets", ["All", "Positive", "Neutral", "Negative"])
    if sentiment_filter != "All":
        filtered_df = df_sentiment[df_sentiment["Avg Sentiment Score"] > 0] if sentiment_filter == "Positive" else \
                      df_sentiment[df_sentiment["Avg Sentiment Score"] < 0] if sentiment_filter == "Negative" else \
                      df_sentiment[df_sentiment["Avg Sentiment Score"] == 0]
    else:
        filtered_df = df_sentiment

    # Display Filtered Tweets
    if not filtered_df.empty:
        st.subheader(f"📝 {sentiment_filter} Tweets")
        st.write(filtered_df[["Date", "Tweet", "Avg Sentiment Score"]])
    else:
        st.warning("⚠️ No tweets available for this category.")

else:
    st.warning("⚠️ Sentiment data not available!")
