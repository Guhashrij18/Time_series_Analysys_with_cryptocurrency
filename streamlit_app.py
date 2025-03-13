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

    # ---- Live Bitcoin Price ----
    st.subheader("💰 Live Bitcoin Price")
    
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
        st.error("Failed to fetch live price. Try again later.")

    # ---- Bitcoin Price Data Table ----
    st.subheader("📊 Bitcoin Price Data (Last 100 Days)")
    st.dataframe(df_prices.tail(100))

    # ---- Moving Averages ----
    st.subheader("📉 Bitcoin Price with Moving Averages")
    df_prices["SMA_50"] = df_prices["Price"].rolling(window=50).mean()
    df_prices["EMA_20"] = df_prices["Price"].ewm(span=20, adjust=False).mean()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_prices.index, df_prices["Price"], label="Bitcoin Price", color="blue")
    ax.plot(df_prices.index, df_prices["SMA_50"], label="50-day SMA", linestyle="dashed", color="red")
    ax.plot(df_prices.index, df_prices["EMA_20"], label="20-day EMA", linestyle="dashed", color="green")
    ax.legend()
    st.pyplot(fig)

    # ---- Forecasting Period Selection ----
    st.subheader("📅 Choose Forecasting Period")
    forecast_days = st.slider("Select number of days to forecast", min_value=30, max_value=180, step=30, value=60)
    
    df_arima = df_arima.head(forecast_days)
    df_lstm = df_lstm.head(forecast_days)
    df_prophet = df_prophet.head(forecast_days)

    # ---- Forecasting Graphs ----
    st.subheader(f"🔮 Forecasting for {forecast_days} Days")

    models = {"ARIMA": df_arima, "LSTM": df_lstm, "Prophet": df_prophet}
    for model_name, df_model in models.items():
        st.subheader(f"{model_name} Model Prediction")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_prices.index[-100:], df_prices["Price"].iloc[-100:], label="Actual Price", color="blue")
        ax.plot(df_model.index[-100:], df_model["Forecast"].iloc[-100:], label=f"{model_name} Forecast", linestyle="dashed")
        ax.legend()
        st.pyplot(fig)

    # ---- Crypto Market Sentiment Analysis ----
    st.subheader("📢 Overall Crypto Market Sentiment")

    avg_sentiment = df_sentiment["Sentiment Score"].mean()

    if avg_sentiment > 0:
        st.markdown(f"🟢 **Positive Market Sentiment** (Score: {avg_sentiment:.2f})")
    elif avg_sentiment < 0:
        st.markdown(f"🔴 **Negative Market Sentiment** (Score: {avg_sentiment:.2f})")
    else:
        st.markdown(f"⚪ **Neutral Market Sentiment** (Score: {avg_sentiment:.2f})")

    # ---- Bar Chart for Sentiment Score Distribution ----
    st.subheader("📊 Sentiment Score Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))
    df_sentiment["Sentiment Score"].hist(bins=20, edgecolor="black", alpha=0.7, ax=ax)
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Sentiment Scores")

    st.pyplot(fig)

    # ---- AI Chatbot for Crypto Analysis ----
    st.subheader("🤖 Bitcoin AI Chatbot")
    OPENAI_API_KEY = "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"  # Replace this with your API key

    def ask_chatbot(prompt):
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]

    user_input = st.text_input("Ask anything about Bitcoin...")
    if user_input:
        answer = ask_chatbot(user_input)
        st.write(answer)

except FileNotFoundError as e:
    st.error(f"Error loading data: {e}")
