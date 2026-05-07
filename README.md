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
Evaluation on out-of-sample data for high-volatility assets.

### Bitcoin (BTC-USD)
- Directional Accuracy: 51.70%
- Strategy Cumulative Return: 1.12x
- Buy & Hold Return: 0.90x

### GameStop (GME)
- Directional Accuracy: 53.47%
- Strategy Cumulative Return: 1.13x
- Buy & Hold Return: 1.14x

## Data
The model downloads historical OHLCV data using the yfinance library.
