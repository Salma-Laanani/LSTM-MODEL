"""
Stock price prediction Streamlit app using LSTM and yfinance

Requirements:
- streamlit
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow (for keras)

Run:
streamlit run stock_lstm_streamlit.py

This single-file app will:
- download historical data using yfinance
- preprocess prices (opening or closing price selectable)
- train an LSTM model
- display predictions and simple evaluation
- allow forecasting next N days
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import datetime
import os

st.set_page_config(page_title="Stock LSTM Predictor", layout="wide")

# ---------------------- Utility functions ----------------------
@st.cache_data
def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data found for ticker/date range. Check the ticker symbol and dates.")
    return df


def create_sequences(values: np.ndarray, seq_len: int):
    """Create sequences for LSTM: (n_samples, seq_len, features) and targets"""
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape, units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def forecast_future(model, last_sequence, n_steps, scaler):
    """Forecast n_steps ahead using iterative predictions."""
    seq = last_sequence.copy()
    preds = []
    for _ in range(n_steps):
        pred = model.predict(seq.reshape(1, seq.shape[0], seq.shape[1]), verbose=0)
        preds.append(pred[0, 0])
        # append prediction and slide window
        seq = np.vstack([seq[1:], [[pred[0, 0]]]])
    # inverse transform
    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds)
    return preds_inv.flatten()


# ---------------------- Streamlit app ----------------------
st.title("📈 Stock Price Prediction with LSTM")
st.markdown("This app downloads historical stock prices using yfinance, trains a simple LSTM on selected prices, and forecasts future prices.")

with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker (e.g. AAPL)", value="AAPL")
    start_date = st.date_input("Start date", value=datetime.date.today() - datetime.timedelta(days=365*3))
    end_date = st.date_input("End date", value=datetime.date.today())
    price_type = st.selectbox("Price type", ["Open", "Close"], index=0)
    seq_len = st.number_input("Sequence length (timesteps)", min_value=5, max_value=365, value=60)
    test_ratio = st.slider("Test set ratio", min_value=0.05, max_value=0.5, value=0.2)
    units = st.number_input("LSTM units", min_value=8, max_value=512, value=50)
    dropout = st.slider("Dropout", min_value=0.0, max_value=0.9, value=0.2)
    epochs = st.number_input("Epochs", min_value=1, max_value=500, value=30)
    batch_size = st.number_input("Batch size", min_value=1, max_value=1024, value=32)
    forecast_days = st.number_input("Forecast horizon (days)", min_value=1, max_value=90, value=7)
    train_button = st.button("Train / (Re)fit model")

# Main panel
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Price data and model outputs")
    try:
        df = download_data(ticker, start_date.strftime("%Y-%m-%d"), (end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

    st.write(df.tail())

    prices = df[[price_type]].copy()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.values)

    # create sequences
    X, y = create_sequences(scaled, seq_len)
    if X.size == 0:
        st.error("Not enough data to create sequences with the chosen sequence length. Try reducing the sequence length or increasing the date range.")
        st.stop()

    # train/test split
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    st.write(f"Samples: total={len(X)}, train={len(X_train)}, test={len(X_test)}")

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), units=units, dropout=dropout)

    if train_button:
        with st.spinner("Training model — this may take a while depending on epochs and data size..."):
            es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[es],
                verbose=1,
            )
            st.success("Training finished")
            # show training loss curve
            fig, ax = plt.subplots()
            ax.plot(history.history.get("loss", []), label="train_loss")
            ax.plot(history.history.get("val_loss", []), label="val_loss")
            ax.set_title("Training loss")
            ax.legend()
            st.pyplot(fig)

            # Save model to local file so user can reuse
            model_path = f"lstm_{ticker}_{price_type}.h5"
            model.save(model_path)
            st.write(f"Model saved to `{model_path}` in current working directory.")

    # If there's a saved model file present, offer to load it
    model_file = f"lstm_{ticker}_{price_type}.h5"
    if os.path.exists(model_file) and not train_button:
        if st.checkbox("Load existing model instead of training", value=False):
            from tensorflow.keras.models import load_model

            try:
                model = load_model(model_file)
                st.info(f"Loaded model from {model_file}")
            except Exception as e:
                st.error(f"Failed to load model: {e}")

    # predictions on test set
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_inv = scaler.inverse_transform(y_test)

    # align indices with original data for plotting
    test_start_idx = seq_len + split_idx
    dates = prices.index[test_start_idx : test_start_idx + len(y_test_inv)]
    pred_series = pd.Series(y_pred.flatten(), index=dates)
    true_series = pd.Series(y_test_inv.flatten(), index=dates)

    # show metrics
    rmse = np.sqrt(mean_squared_error(true_series, pred_series))
    mae = mean_absolute_error(true_series, pred_series)
    st.metric("Test RMSE", f"{rmse:.4f}")
    st.metric("Test MAE", f"{mae:.4f}")

    # Plot actual vs predicted
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(prices.index, prices[price_type], label=f"Actual {price_type}")
    ax2.plot(pred_series.index, pred_series.values, label="Predicted (test)")
    ax2.set_title(f"{ticker} {price_type} Price — Actual vs Predicted (test)")
    ax2.legend()
    st.pyplot(fig2)

    # Forecast future
    if st.button("Forecast future"):
        last_seq = scaled[-seq_len:]
        preds = forecast_future(model, last_seq, forecast_days, scaler)
        future_dates = [prices.index[-1] + datetime.timedelta(days=i + 1) for i in range(forecast_days)]
        forecast_df = pd.DataFrame({"Forecast": preds}, index=future_dates)

        st.subheader(f"{forecast_days}-day forecast")
        st.write(forecast_df)

        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(prices.index[-(seq_len * 2):], prices[price_type].iloc[-(seq_len * 2):], label="Recent Actual")
        ax3.plot(forecast_df.index, forecast_df.values, marker="o", linestyle="--", label="Forecast")
        ax3.set_title(f"Forecast next {forecast_days} days for {ticker} ({price_type})")
        ax3.legend()
        st.pyplot(fig3)

with col2:
    st.subheader("Data explorer")
    st.write("Summary statistics:")
    st.write(prices.describe())
    st.write("Volume (last 10 rows):")
    st.write(df["Volume"].tail(10))

    st.subheader("Download")
    csv = df.to_csv().encode("utf-8")
    st.download_button(label="Download raw data (CSV)", data=csv, file_name=f"{ticker}_data.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("**Notes & tips**")
    st.markdown("- This is a simple demonstration and not financial advice.\n- LSTM on raw open or close prices is a naive model — consider adding features (technical indicators, volumes, returns) and more robust validation for production use.\n- Training in Streamlit may be slow; consider training offline and loading a saved model for fast experimentation.")

st.sidebar.markdown("---")
st.sidebar.write("Built with ❤️ — modify hyperparameters and try different tickers or longer histories.")