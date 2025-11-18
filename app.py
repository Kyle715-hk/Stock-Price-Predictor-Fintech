import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load trained model and scaler
@st.cache_resource
def load_assets():
    model = load_model('stock_predictor_lstm_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# Functions from previous steps (adapted for app)
def fetch_data(ticker, start='2015-11-14', end='2025-11-15'):
    """Fetch stock, VIX, rates data using yfinance (fallback for simplicity in app)"""
    stock_raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    vix_raw = yf.download('^VIX', start=start, end=end, progress=False, auto_adjust=False)
    rates_raw = yf.download('^TNX', start=start, end=end, progress=False, auto_adjust=False)

    combined = pd.concat([
        stock_raw['Close'].rename('Stock_Close'),
        vix_raw['Close'].rename('VIX'),
        rates_raw['Close'].rename('Interest_Rate')
    ], axis=1).ffill().dropna()  # Simple merge with forward fill

    return combined

def preprocess_data(df, seq_length=60):
    df['MA_50'] = df['Stock_Close'].rolling(window=50).mean()
    df = df.dropna()

    features = ['Stock_Close', 'MA_50', 'VIX', 'Interest_Rate']
    scaled_data = scaler.transform(df[features])

    X = []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
    return np.array(X)

# Streamlit App
st.title('Fintech Stock Predictor: Simulate Revolut Investing Tools')
st.markdown("""
This app forecasts stock prices using an LSTM model, with sensitivity to volatility and rates.
Inspired by Revolut's user-facing investment features for real-time decisions.
""")

# User inputs
ticker = st.text_input('Enter Stock Ticker (e.g., TSLA or AAPL)', value='TSLA')
if st.button('Run Prediction'):

    with st.spinner('Fetching data and predicting...'):
        try:
            # Fetch and preprocess
            df = fetch_data(ticker)
            if df.empty:
                raise ValueError("No data fetched. Check ticker.")

            X = preprocess_data(df)
            if len(X) == 0:
                raise ValueError("Insufficient data for sequences.")

            # Predict (use last sequence for next-day forecast)
            last_seq = X[-1:]  # Shape: (1, 60, 4)
            pred_scaled = model.predict(last_seq)

            # Inverse scale
            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[0, 0] = pred_scaled[0]
            pred_price = scaler.inverse_transform(dummy)[0, 0]

            last_price = df['Stock_Close'].iloc[-1]
            st.success(f'Predicted Next-Day Price for {ticker}: ${pred_price:.2f} (Current: ${last_price:.2f})')

            # Quick evaluation on historical test (simplified)
            predictions = model.predict(X)
            dummy = np.zeros((len(predictions), scaler.n_features_in_))
            dummy[:, 0] = predictions.flatten()
            preds = scaler.inverse_transform(dummy)[:, 0]

            actual = df['Stock_Close'].iloc[60:].values  # Align with sequences
            rmse = np.sqrt(mean_squared_error(actual, preds))
            st.info(f'Historical RMSE: {rmse:.2f}')

            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df.index[60:], actual, label='Actual')
            ax.plot(df.index[60:], preds, label='Predicted', linestyle='--')
            ax.set_title(f'{ticker} Price Prediction')
            ax.legend()
            st.pyplot(fig)

            # Sensitivity (simplified for app: show one scenario)
            st.subheader('Sensitivity Example (10% Volatility Noise)')
            vol_level = 0.10
            X_vol = X.copy()
            X_vol[:, :, 2] += np.random.normal(0, vol_level, X_vol[:, :, 2].shape) * X_vol[:, :, 2]  # Perturb VIX

            preds_vol = model.predict(X_vol)
            dummy[:, 0] = preds_vol.flatten()
            preds_vol_inv = scaler.inverse_transform(dummy)[:, 0]

            rmse_vol = np.sqrt(mean_squared_error(actual, preds_vol_inv))
            st.write(f'RMSE under 10% volatility: {rmse_vol:.2f}')

        except Exception as e:
            st.error(f'Error: {str(e)}. Try another ticker or check data availability.')