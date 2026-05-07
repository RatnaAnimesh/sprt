import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import pandas as pd
from train_transformer import GeneralPriceTransformer, StockDataset
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def run_evaluation(ticker="BTC-USD", lookback=30):
    print(f"--- Hard Asset Evaluation: {ticker} ---")
    
    # 1. Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2) # 2 years
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        print("Failed to download data.")
        return

    # 2. Prepare Data
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    
    # Split into Train (80%) and Test (20%) - Sequential split for time series
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx - lookback:] # overlap for lookback
    
    # --- Training Phase ---
    print(f"Training on historical data ({len(train_data)} points)...")
    X_train, y_train = [], []
    for i in range(len(train_data) - lookback):
        window = train_data[i:i+lookback].copy()
        target = train_data[i+lookback, 3]
        baseline = window[0, 3]
        X_train.append(window / baseline)
        y_train.append(target / baseline)
    
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32).view(-1, 1)
    
    model = GeneralPriceTransformer(input_dim=5)
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # Mini training loop
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.6f}")

    # --- Testing Phase (Forward Test) ---
    print(f"\nForward testing on out-of-sample data ({len(test_data)-lookback} points)...")
    model.eval()
    X_test, y_test_actual, dates = [], [], []
    baselines = []
    
    test_dates = df.index[split_idx:]
    
    for i in range(len(test_data) - lookback):
        window = test_data[i:i+lookback].copy()
        actual = test_data[i+lookback, 3]
        baseline = window[0, 3]
        
        X_test.append(window / baseline)
        y_test_actual.append(actual)
        baselines.append(baseline)
        dates.append(test_dates[i])
        
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    
    with torch.no_grad():
        preds_norm = model(X_test).numpy().flatten()
    
    # Denormalize
    preds = preds_norm * np.array(baselines)
    y_test_actual = np.array(y_test_actual)
    
    # --- Metrics ---
    # Directional Accuracy: Did we correctly predict if price goes up or down relative to current?
    current_prices = test_data[lookback-1:-1, 3]
    actual_dir = np.sign(y_test_actual - current_prices)
    pred_dir = np.sign(preds - current_prices)
    dir_acc = np.mean(actual_dir == pred_dir)
    
    print(f"\nDirectional Accuracy: {dir_acc:.2%}")
    
    # Simple Strategy PnL: Long if pred > current, else Short (normalized to 1.0 start)
    returns = (y_test_actual[1:] - y_test_actual[:-1]) / y_test_actual[:-1]
    strat_signals = pred_dir[:-1]
    strat_returns = strat_signals * returns
    cum_strat = np.cumprod(1 + strat_returns)
    cum_bh = np.cumprod(1 + returns)
    
    print(f"Strategy Final Return: {cum_strat[-1]:.2f}x")
    print(f"Buy & Hold Final Return: {cum_bh[-1]:.2f}x")

    # --- Plotting with Seaborn ---
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(14, 7))
    
    plot_df = pd.DataFrame({
        'Date': dates,
        'Actual': y_test_actual,
        'Predicted': preds
    })
    
    sns.lineplot(data=plot_df, x='Date', y='Actual', label='Actual Price', linewidth=2)
    sns.lineplot(data=plot_df, x='Date', y='Predicted', label='Predicted Price', linestyle='--', alpha=0.8)
    
    plt.title(f"Forward Test Evaluation: {ticker}", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ticker}_forward_test.png", dpi=300)
    print(f"Enhanced seaborn plot saved to {ticker}_forward_test.png")

if __name__ == "__main__":
    import torch.optim as optim
    # Bitcoin is notoriously hard to predict due to high volatility and lack of traditional fundamentals
    run_evaluation("BTC-USD")
