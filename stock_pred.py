import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time

# Function to fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1), stock.info['currency']

# Function to preprocess data for LSTM
def preprocess_data(data, look_back):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x, y = [], []
    for i in range(len(scaled_data) - look_back):
        x.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    return x, y, scaler

# Function to create and train the LSTM model
def create_lstm_model(look_back):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to predict future stock prices
def predict_stock_prices(model, x, scaler):
    predicted_prices = model.predict(x)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    return predicted_prices

# Main function
def main():
    # User input for stock symbol and prediction date
    ticker = input("Enter stock symbol (e.g., AAPL): ")
    prediction_date = input("Enter the date for stock price prediction (YYYY-MM-DD): ")

    # Fetch historical stock data
    start_date = '2000-01-01'  # You can adjust the start date as needed
    end_date = prediction_date
    stock_data, currency = get_stock_data(ticker, start_date, end_date)

    # Preprocess data for LSTM
    look_back = 60 # You can adjust the look-back period
    x, y, scaler = preprocess_data(stock_data, look_back)

    # Create and train the LSTM model
    model = create_lstm_model(look_back)
    model.fit(x, y, epochs=10, batch_size=32)

    # Use the model to predict future stock prices
    last_sequence = stock_data[-look_back:]
    last_sequence = scaler.transform(last_sequence.reshape(-1, 1))
    last_sequence = np.reshape(last_sequence, (1, look_back, 1))
    predicted_prices = predict_stock_prices(model, last_sequence, scaler)

    print(f"Prediction for Stock Price of {ticker} on {prediction_date}: {currency} {predicted_prices[0][0]:.2f}")

if __name__ == "__main__":
    while True:
        main()
