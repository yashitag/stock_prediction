# Install necessary packages if not already installed
# !pip install yfinance pandas numpy matplotlib seaborn statsmodels tensorflow sklearn

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Data Collection
stock_data = yf.download('AAPL', start='2015-01-01', end='2023-01-01')
data = stock_data[['Adj Close']].dropna()
data.columns = ['Price']

# Step 2: Data Visualization
plt.figure(figsize=(12, 6))
plt.plot(data, label="Stock Price")
plt.title("Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# Step 3: ARIMA Model for Time Series Prediction
train_data, test_data = data['Price'][:-60], data['Price'][-60:]

# Fit ARIMA model
arima_model = ARIMA(train_data, order=(5, 1, 0))  # You can tune the (p, d, q) parameters
arima_fitted = arima_model.fit()

# Forecast with ARIMA
arima_forecast = arima_fitted.forecast(steps=len(test_data))

# Plot ARIMA Results
plt.figure(figsize=(10, 6))
plt.plot(train_data, label='Training Data')
plt.plot(test_data, label='Actual Stock Price')
plt.plot(test_data.index, arima_forecast, label='ARIMA Predicted Price', color='red')
plt.legend(loc='upper left')
plt.title('ARIMA Model Prediction')
plt.show()

# Step 4: Data Preparation for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

train_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

# Function to create sequences for LSTM input
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Reshape data for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Step 5: Building and Training the LSTM Model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, batch_size=32, epochs=10)

# Step 6: LSTM Prediction and Visualization
lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Prepare the data for visualization
# Prepare the data for visualization
train = data[:train_size]
valid = data[train_size:]

# Adjust valid to match the length of LSTM predictions
valid = valid[-len(lstm_predictions):]
valid['LSTM Predictions'] = lstm_predictions

plt.figure(figsize=(10, 6))
plt.plot(train['Price'], label='Train')
plt.plot(valid['Price'], label='Actual Price')
plt.plot(valid['LSTM Predictions'], label='LSTM Predicted Price')
plt.legend()
plt.title('LSTM Model Prediction')
plt.show()
