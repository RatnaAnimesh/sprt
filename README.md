# Sequential Price Regression Transformer

A transformer-based model for predicting continuous price values from sequential market data. Originally meant for IMC Prosperity 4.

## Features

- Sequential price prediction using transformer architecture
- Data preprocessing pipeline for market CSV files
- Kalman filter implementation for signal smoothing

## Usage

Run the training script with a yfinance ticker (e.g., AAPL, GC=F for Gold, BTC-USD):

```bash
python train_transformer.py AAPL
```

## Backtesting Results

Evaluation on out-of-sample data for high-volatility asset.

### Bitcoin (BTC-USD)

- Directional Accuracy: 50.68%
- Strategy Cumulative Return: 0.82x
- Buy & Hold Return: 0.88x

![Bitcoin Forward Test](BTC-USD_forward_test.png)

### Ethereum (ETH-USD)

- Directional Accuracy: 50.68%
- Strategy Cumulative Return: 0.94x
- Buy & Hold Return: 0.74x

![Ethereum Forward Test](ETH-USD_forward_test.png)

## Data

The model downloads historical OHLCV data using the yfinance library.
