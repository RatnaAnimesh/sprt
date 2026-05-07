import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import math
import os

# --- Model Architecture ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class GeneralPriceTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super(GeneralPriceTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.decoder(x[:, -1, :])

# --- Data Handling ---
class StockDataset(Dataset):
    def __init__(self, ticker, lookback=30, period="2y", interval="1d"):
        print(f"Fetching data for {ticker}...")
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Features: Open, High, Low, Close, Volume
        data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        
        # Normalize (Min-Max per window is better for generalization)
        self.raw_data = data
        self.lookback = lookback
        self.X, self.y = [], []
        
        for i in range(len(data) - lookback):
            window = data[i:i+lookback].copy()
            target = data[i+lookback, 3] # Close price
            
            # Simple normalization: divide by the first Close in the window
            baseline = window[0, 3]
            window_norm = window / baseline
            target_norm = target / baseline
            
            self.X.append(window_norm)
            self.y.append(target_norm)
            
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_generalized_model(ticker="AAPL", epochs=20):
    lookback = 30
    try:
        dataset = StockDataset(ticker, lookback=lookback)
    except Exception as e:
        print(f"Error: {e}")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = GeneralPriceTransformer(input_dim=5)
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print(f"Starting training for {ticker}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in test_loader:
                v_pred = model(bx)
                val_loss += criterion(v_pred, by).item()
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss/len(train_loader):.6f} | Val Loss: {val_loss/len(test_loader):.6f}")

    torch.save(model.state_dict(), f"{ticker}_transformer.pth")
    print(f"Model saved to {ticker}_transformer.pth")

if __name__ == "__main__":
    import sys
    ticker_to_train = "GC=F" # Default Gold Futures
    if len(sys.argv) > 1:
        ticker_to_train = sys.argv[1]
    
    train_generalized_model(ticker_to_train)
