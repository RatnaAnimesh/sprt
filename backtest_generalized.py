import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import pandas as pd
from train_transformer import MultiHorizonTransformer, MultiHorizonDataset, SentimentEngine, DPMLoss
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def run_evaluation(ticker="BTC-USD", lookback=64):
    print(f"--- SOTA Research Evaluation: {ticker} ---")
    
    # 1. Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2) # 2 years
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        print("Failed to download data.")
        return

    # 2. Prepare Data
    data_price = df[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
    
    # --- Real Sentiment Feature (Current yfinance News Only) ---
    # Historical news is unavailable via yfinance; initializing sentiment state as 0.
    # The architecture remains 6-dimensional to support live real-time inference.
    sentiment_feature = np.zeros(len(data_price))
    try:
        tkr = yf.Ticker(ticker)
        headlines = [n['content']['title'] for n in tkr.news if 'content' in n]
        engine = SentimentEngine()
        sentiment_feature[-1] = engine.process_headlines(headlines)
    except:
        pass
    
    data = np.column_stack((data_price, sentiment_feature.astype(np.float32)))
    
    # Split into Train (80%) and Test (20%) - Sequential split for time series
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx - lookback:] # overlap for lookback
    
    # --- Training Phase ---
    print(f"Training on historical data ({len(train_data)} points)...")
    X_train, y_train = [], []
    for i in range(len(train_data) - lookback - 14):
        X_train.append(train_data[i:i+lookback])
        # Current close for return calculation
        c = train_data[i+lookback-1, 3]
        y1 = np.log(train_data[i+lookback, 3] / c)
        y3 = np.log(train_data[i+lookback+2, 3] / c)
        y7 = np.log(train_data[i+lookback+6, 3] / c)
        y14 = np.log(train_data[i+lookback+13, 3] / c)
        y_train.append([y1, y3, y7, y14])
    
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
    
    model = MultiHorizonTransformer(input_dim=6, lookback=lookback)
    criterion = DPMLoss() # Breaking mirroring with momentum loss
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    # Training loop
    for epoch in range(200): # Increased slightly for multi-horizon
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1:02d} | Multi-Horizon Loss: {loss.item():.6f}")

    # --- Testing Phase (Forward Test) ---
    print(f"\nForward testing on out-of-sample data ({len(data) - split_idx} points)...")
    model.eval()
    X_test, y_test_actual, dates = [], [], []
    
    test_dates = df.index[split_idx:]
    
    for i in range(len(test_data) - lookback - 14):
        X_test.append(test_data[i:i+lookback])
        y_test_actual.append(test_data[i+lookback, 3]) # Still track absolute price for plotting
        dates.append(test_dates[i])
        
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    
    with torch.no_grad():
        preds_returns = model(X_test).numpy()
        # Convert first horizon (t+1) return back to price for plotting
        current_closes = np.array([test_data[i+lookback-1, 3] for i in range(len(preds_returns))])
        preds = current_closes * np.exp(preds_returns[:, 0])
    
    y_test_actual = np.array(y_test_actual)
    
    # Directional Accuracy based on first horizon prediction
    current_prices = np.array([test_data[i+lookback-1, 3] for i in range(len(preds))])
    actual_dir = np.sign(y_test_actual - current_prices)
    pred_dir = np.sign(preds_returns[:, 0]) # Positive return means up
    dir_acc = np.mean(actual_dir == pred_dir)
    
    print(f"\nDirectional Accuracy (t+1): {dir_acc:.2%}")
    
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # 1. Price Plot
    plot_df = pd.DataFrame({
        'Date': dates,
        'Actual': y_test_actual,
        'Predicted': preds
    })
    sns.lineplot(data=plot_df, x='Date', y='Actual', label='Actual Price', linewidth=2, ax=ax1)
    sns.lineplot(data=plot_df, x='Date', y='Predicted', label='Predicted Price', linestyle='--', alpha=0.8, ax=ax1)
    ax1.set_title(f"Price Prediction Forward Test: {ticker}", fontsize=16)
    ax1.set_ylabel("Price (USD)", fontsize=12)
    
    # 2. Cumulative Return Plot
    # Returns aligned with dates[1:]
    return_df = pd.DataFrame({
        'Date': dates[1:],
        'Strategy': cum_strat,
        'Buy & Hold': cum_bh
    })
    sns.lineplot(data=return_df, x='Date', y='Strategy', label='Model Strategy', linewidth=2, color='green', ax=ax2)
    sns.lineplot(data=return_df, x='Date', y='Buy & Hold', label='Buy & Hold', linewidth=2, color='gray', alpha=0.6, ax=ax2)
    ax2.set_title(f"Cumulative Return Comparison", fontsize=16)
    ax2.set_ylabel("Growth (1.0 = Start)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{ticker}_forward_test.png", dpi=300)
    print(f"Comprehensive performance plot saved to {ticker}_forward_test.png")
    
    # --- Audit Export: Check for Mirroring ---
    audit_df = pd.DataFrame({
        'Date': dates,
        'Actual_Price': y_test_actual,
        'Predicted_Price': preds,
        'Predicted_Return_t1': preds_returns[:, 0]
    })
    audit_df.to_csv(f"audit_results_{ticker}.csv", index=False)
    print(f"Audit CSV saved to audit_results_{ticker}.csv")

if __name__ == "__main__":
    import torch.optim as optim
    run_evaluation("BTC-USD")
    run_evaluation("ETH-USD")
    
    # --- LIVE SENTIMENT PREDICTION HOOK ---
    print("\n" + "="*30)
    print("LIVE REAL-TIME SENTIMENT PREDICTION")
    print("="*30)
    for tkr in ["BTC-USD", "ETH-USD"]:
        ticker = yf.Ticker(tkr)
        # yfinance news structure: list of dicts with 'content' -> 'title'
        headlines = []
        if ticker.news:
            for n in ticker.news:
                if 'content' in n and 'title' in n['content']:
                    headlines.append(n['content']['title'])
        
        engine = SentimentEngine()
        score = engine.process_headlines(headlines)
        print(f"\nAsset: {tkr}")
        print(f"Recent Headlines Found: {len(headlines)}")
        print(f"Neural Sentiment State: {score:.4f} " + (" [POLARIZED RESET]" if abs(score) > 0.6 else " [STABLE DECAY]"))
        if headlines: 
            print(f"Latest Impact Headline: {headlines[0]}")

