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

Evaluation on out-of-sample data using the **Multi-Horizon Research Transformer** (Log-Return Target). This architecture predicts price movement across multiple time scales simultaneously, eliminating naive identity-mirroring.

### Bitcoin (BTC-USD)

- Directional Accuracy (t+1): 51.52%
- Strategy Cumulative Return: 0.65x
- Buy & Hold Return: 0.87x

![Bitcoin Forward Test](BTC-USD_forward_test.png)

### Ethereum (ETH-USD)

- Directional Accuracy (t+1): 53.03%
- Strategy Cumulative Return: 1.00x
- Buy & Hold Return: 0.75x

![Ethereum Forward Test](ETH-USD_forward_test.png)

## Data

The model downloads historical OHLCV data using the yfinance library.
