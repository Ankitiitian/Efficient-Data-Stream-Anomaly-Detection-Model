import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import load_model
import yfinance as yf

def synthetic_stock_price_stream(mean=100, seasonal_variation=5, noise_level=1.0, trend=0.1, volatility=1.0):
    t = 0
    price = mean
    while True:
        seasonal_effect = seasonal_variation * np.sin(2 * np.pi * t / 50)
        trend_effect = trend * t
        noise = np.random.normal(0, noise_level)
        jump = np.random.normal(0, volatility) if np.random.rand() < 0.1 else 0
        price += noise + seasonal_effect + trend_effect + jump
        yield price
        t += 1

def fetch_real_stock_data(ticker='VOW3', period='1y', interval='1d'):
    stock_data = yf.download(ticker, period=period, interval=interval)
    return stock_data['Close'].values

def build_lstm_autoencoder(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape, return_sequences=False))
    model.add(RepeatVector(input_shape[0]))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[1])))
    model.compile(optimizer='adam', loss='mse')
    return model

def collect_normal_data(data_stream, num_samples=1000, time_steps=10):
    X_train = []
    window = []
    
    for i, data_point in enumerate(data_stream):
        if data_point is not None:
            window.append([data_point])
            if len(window) == time_steps:
                X_train.append(window)
                window = []
            if len(X_train) == num_samples:
                break
    return np.array(X_train)

def calculate_dynamic_threshold(model, window, multiplier=3.0):
    window_array = np.array(window).reshape(1, len(window), 1)
    reconstructed = model.predict(window_array)
    error = np.mean(np.abs(window_array - reconstructed))
    return error + multiplier * np.std(error)

def real_time_plot_with_dynamic_threshold(data_stream, model, time_steps=10, plot_interval=10, smoothing_window=60):
    window = deque(maxlen=time_steps)
    error_window = deque(maxlen=smoothing_window)
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.ion()

    x_data, y_data = [], []
    anomaly_x, anomaly_y = [], []
    anomaly_ranges = []
    prediction_y = []
    smoothed_prediction_y = []
    thresholds = []

    anomaly_start = None

    for idx, data_point in enumerate(data_stream):
        if data_point is not None:
            window.append([data_point])

            if len(window) == time_steps:
                window_array = np.array(window).reshape(1, time_steps, 1)
                reconstructed = model.predict(window_array)
                error = np.mean(np.abs(window_array - reconstructed))
                error_window.append(error)

                # Calculate dynamic threshold
                dynamic_threshold = np.mean(error_window) + 3 * np.std(error_window)
                thresholds.append(dynamic_threshold)

                x_data.append(idx)
                y_data.append(data_point)
                prediction_y.append(reconstructed[0, -1, 0])

                # Detect anomaly and update anomaly ranges
                if error > dynamic_threshold:
                    anomaly_x.append(idx)
                    anomaly_y.append(data_point)
                    if anomaly_start is None:
                        anomaly_start = idx
                else:
                    if anomaly_start is not None:
                        anomaly_ranges.append((anomaly_start, idx))
                        anomaly_start = None

                # Calculate smoothed prediction
                if len(prediction_y) >= smoothing_window:
                    smoothed_value = np.mean(prediction_y[-smoothing_window:])
                else:
                    smoothed_value = np.mean(prediction_y)
                smoothed_prediction_y.append(smoothed_value)

                if idx % plot_interval == 0:
                    ax.clear()
                    ax.plot(x_data, y_data, color='blue', label='Actual Stock Prices')
                    ax.plot(x_data, smoothed_prediction_y, color='green', label='Smoothed Predicted Prices', alpha=0.7)
                    ax.plot(x_data, thresholds, color='orange', linestyle='--', label='Dynamic Threshold')
                    
                    # Plot vertical lines for anomaly ranges
                    for start, end in anomaly_ranges:
                        ax.axvspan(start, end, color='red', alpha=0.3)
                    
                    # Plot red points for individual anomalies
                    ax.scatter(anomaly_x, anomaly_y, color='red', marker='o', label='Anomalies')
                    
                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel('Stock Price')
                    ax.set_title('Real-Time Stock Prices with Dynamic Threshold and Anomaly Detection')
                    ax.legend()
                    plt.pause(0.01)

    # Handle case where anomaly continues to the end
    if anomaly_start is not None:
        anomaly_ranges.append((anomaly_start, len(x_data) - 1))

    plt.savefig('/home/joy/Desktop/Cobblestone/stock_plot_dynamic_threshold.png')
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    # Fetch real stock prices
    real_stock_prices = fetch_real_stock_data(ticker='NVDA', period='10y', interval='1d')
    stream = (price for price in real_stock_prices)
    
    input_shape = (10, 1)
    X_train = collect_normal_data(stream, num_samples=3000, time_steps=10)

    # Build and train the LSTM Autoencoder Model
    lstm_autoencoder = build_lstm_autoencoder(input_shape)
    split = int(0.8 * len(X_train))
    X_train_data = X_train[:split]
    X_val_data = X_train[split:]

    lstm_autoencoder.fit(
        X_train_data, X_train_data,
        epochs=150,
        batch_size=64,
        validation_data=(X_val_data, X_val_data),
        shuffle=True
    )

    # Save and load the model
    lstm_autoencoder.save("lstm_autoencoder_model.keras")
    lstm_autoencoder = load_model("lstm_autoencoder_model.keras")

    # Reset the stream for real-time detection
    real_stock_prices = fetch_real_stock_data(ticker='NVDA', period='10y', interval='1d')
    stream = (price for price in real_stock_prices)

    # Run real-time analysis with dynamic threshold
    real_time_plot_with_dynamic_threshold(stream, lstm_autoencoder, time_steps=10, smoothing_window=60)