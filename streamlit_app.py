import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# ---- Load Bitcoin Price Data ----
try:
    # Load Bitcoin Price Data (assuming you have a CSV file with a 'Date' and 'Price' column)
    df_prices = pd.read_csv("bitcoin_prices.csv", parse_dates=["Date"], index_col="Date")

    # Filter the last 100 days of data
    df_last_100_days = df_prices.tail(100)  # Get the last 100 days of price data

    # ---- Streamlit UI ----
    st.title("📈 Cryptocurrency Price Forecasting & Sentiment Analysis")
    st.write("Analyze Bitcoin trends using ARIMA, LSTM, Prophet, and sentiment analysis from Twitter.")

    # ---- Current Bitcoin Price ----
    st.subheader("Current Bitcoin Price (USD)")

    # Fetch live Bitcoin price using your API (this will be cached for 5 minutes)
    current_bitcoin_price = get_current_bitcoin_price()

    # Display live Bitcoin price in default color, center-aligned
    st.markdown(f"<h2 style='text-align: center; font-weight: bold;'>${current_bitcoin_price:,.2f}</h2>", unsafe_allow_html=True)

    # ---- Bitcoin Price Trend for Last 100 Days ----
    st.subheader("Bitcoin Price Trend (Last 100 Days)")

    # Line chart for the last 100 days of Bitcoin price
    st.line_chart(df_last_100_days["Price"])

    # ---- Bitcoin Price Data ----
    st.subheader("Bitcoin Price Data")
    st.write("Here is the raw Bitcoin price data for the past 100 days:")
    st.dataframe(df_last_100_days.tail())  # Show last few rows of the filtered price data

except FileNotFoundError as e:
    st.error(f"Error loading data: {e}")

