import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import openai
import os
from dotenv import load_dotenv

# ---- Load Environment Variables ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- Load Data Function ----
def load_data(filename, parse_dates=True):
    """Loads a CSV file into a DataFrame with error handling."""
    try:
        df = pd.read_csv(filename)

        # Special handling for crypto_sentiment.csv (No Date column)
        if filename == "crypto_sentiment.csv":
            return df  # Return as-is

        # Check if "Date" column exists for other files
        if "Date" not in df.columns:
            st.error(f"⚠️ `{filename}` does not have a 'Date' column! Available columns: {list(df.columns)}")
            return None

        df["Date"] = pd.to_datetime(df["Date"])  # Convert to datetime
        df.set_index("Date", inplace=True)  # Set Date as index
        return df

    except FileNotFoundError:
        st.error(f"❌ Error: `{filename}` not found! Please check the file path.")
        return None
    except Exception as e:
        st.error(f"❌ Error while loading `{filename}`: {str(e)}")
        return None

# ---- Load All Data ----
df_prices = load_data("bitcoin_prices.csv")
df_arima = load_data("arima_forecast.csv")
df_lstm = load_data("lstm_forecast.csv")
df_prophet = load_data("prophet_forecast.csv")
df_sentiment = load_data("crypto_sentiment.csv", parse_dates=False)  # No Date column

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
        response.raise_for_status()  # Raise error for bad status code
        return response.json()["bitcoin"]["usd"]
    except requests.RequestException:
        return None

live_price = get_live_price()
if live_price:
    st.metric(label="Current Bitcoin Price (USD)", value=f"${live_price}")
else:
    st.error("⚠️ Failed to fetch live price. Try again later.")

# ---- Bitcoin Price Data Table ----
if df_prices is not None:
    st.subheader("📋 Bitcoin Price Data (Last 100 Days)")
    st.dataframe(df_prices.tail(100))

    # ---- Bitcoin Price Trend ----
    st.subheader("📊 Bitcoin Price Trend")
    st.line_chart(df_prices["Price"])

    # ---- Forecasting Plots ----
    def plot_forecast(actual_df, forecast_df, title, color):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(actual_df.index[-100:], actual_df["Price"].iloc[-100:], label="Actual Price", color="blue")
        ax.plot(forecast_df.index[-100:], forecast_df["Forecast"].iloc[-100:], label=f"{title} Forecast", linestyle="dashed", color=color)
        ax.legend()
        st.pyplot(fig)

    if df_arima is not None:
        st.subheader("🔮 ARIMA Model Prediction")
        plot_forecast(df_prices, df_arima, "ARIMA", "red")

    if df_lstm is not None:
        st.subheader("🤖 LSTM Model Prediction")
        plot_forecast(df_prices, df_lstm, "LSTM", "green")

    if df_prophet is not None:
        st.subheader("🔥 Prophet Model Prediction")
        plot_forecast(df_prices, df_prophet, "Prophet", "purple")

# ---- Sentiment Analysis ----
if df_sentiment is not None:
    st.subheader("📢 Crypto Market Sentiment Analysis")

    # Show Sentiment Data
    st.subheader("🔍 Sentiment Data Preview")
    st.write(df_sentiment.head())  

    # Show Sentiment Distribution Chart
    positive_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] > 0])
    neutral_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] == 0])
    negative_tweets = len(df_sentiment[df_sentiment["Sentiment Score"] < 0])

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

# ---- AI Chatbot for Bitcoin Analysis ----
st.subheader("🤖 Bitcoin AI Chatbot")

if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key is missing! Set it as an environment variable or in a `.env` file.")
else:
    def ask_chatbot(prompt):
        """Function to query OpenAI's chatbot."""
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = openai.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content 
        except Exception as e:
            return f"⚠️ OpenAI API Error: {str(e)}"

    # Get user input for chatbot
    user_input = st.text_input("💬 Ask anything about Bitcoin...")
    if user_input:
        answer = ask_chatbot(user_input)
        st.write(answer)
