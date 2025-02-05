import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import plotly.graph_objects as go

# Function to calculate RSI
def calculate_rsi(close_prices, window=14):
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(close_prices, window=20, num_std_dev=2):
    rolling_mean = close_prices.rolling(window).mean()
    rolling_std = close_prices.rolling(window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

import streamlit as st

# Streamlit app setup
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Title with Emoji
st.title("📉 Stock Price Predictor and Analyzer")
# Custom CSS to enhance the appearance
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #e8f0fe;
    }
    .stDateInput>div>div>input {
        background-color: #e8f0fe;
    }
    .stSelectbox>div>div>div>div {
        background-color: #e8f0fe;
    }
    .stCheckbox>div>div>div>input {
        accent-color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Sidebar inputs
st.sidebar.header("Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
model_type = st.sidebar.selectbox("Choose Prediction Model", ["Linear Regression", "Polynomial Regression"])
degree = st.sidebar.slider("Polynomial Degree (if selected)", 2, 5, 2)
show_rsi = st.sidebar.checkbox("Show RSI")
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands")
show_predictions = st.sidebar.checkbox("Show Predictions", value=True)

# Validation for date inputs
if start_date >= end_date:
    st.error("Start date must be earlier than the end date.")
else:
    if st.sidebar.button("Fetch Data"):
        # Fetch stock data
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            stock_data = pd.DataFrame()  # Define stock_data as an empty DataFrame in case of error

        if stock_data.empty:
            st.warning("No data found for the specified ticker and date range. Please try again.")
        else:
            # Display raw data
            st.subheader(f"Stock Data for {ticker.upper()}")
            st.write(stock_data)

            # Stock Price Chart
            st.subheader("📉 Stock Price Chart") # This is the subheader for the chart section
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name='Candlestick'
                ))
            st.plotly_chart(fig, use_container_width=True)

            # Bollinger Bands Calculation
            if show_bollinger:
                st.subheader("Bollinger Bands")
                upper_band, lower_band = calculate_bollinger_bands(stock_data['Close'])
                fig_bollinger = go.Figure()
                fig_bollinger.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
                fig_bollinger.add_trace(go.Scatter(x=stock_data.index, y=upper_band, mode='lines', name='Upper Band', line=dict(dash='dot')))
                fig_bollinger.add_trace(go.Scatter(x=stock_data.index, y=lower_band, mode='lines', name='Lower Band', line=dict(dash='dot')))
                st.plotly_chart(fig_bollinger, use_container_width=True)

            # RSI Calculation
            if show_rsi:
                st.subheader("Relative Strength Index (RSI)")
                stock_data['RSI'] = calculate_rsi(stock_data['Close'])
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'))
                fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold")
                st.plotly_chart(fig_rsi, use_container_width=True)

            # Stock Price Predictions
            if show_predictions:
                st.subheader("📉 Stock Price Predictions")  # This is the subheader for the prediction section
                stock_data['Prev Close'] = stock_data['Close'].shift(1)
                stock_data.dropna(inplace=True)

                X = stock_data[['Prev Close']]
                y = stock_data['Close']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                if model_type == "Linear Regression":
                    model = LinearRegression()
                else:
                    poly = PolynomialFeatures(degree=degree)
                    X_train = poly.fit_transform(X_train)
                    X_test = poly.transform(X_test)
                    model = LinearRegression()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write(f"R² Score: {r2:.2f}")

                # Plot predictions
                results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}, index=y_test.index)
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=results.index, y=results['Actual'], mode='lines', name='Actual Price'))
                fig_pred.add_trace(go.Scatter(x=results.index, y=results['Predicted'], mode='lines', name='Predicted Price'))
                st.plotly_chart(fig_pred, use_container_width=True)
