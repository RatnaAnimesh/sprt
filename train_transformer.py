import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yfinance as yf
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import math
import os

# --- Reversible Instance Normalization (RevIN) ---
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        """
        Kim et al. (2021) - Normalizes instance statistics to handle distribution shift.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.stdev + self.mean
        return x

# --- PatchTST Style Architecture ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ResearchPriceTransformer(nn.Module):
    def __init__(self, input_dim, lookback, patch_size=8, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        """
        PatchTST-inspired Transformer using RevIN and local patching.
        """
        super(ResearchPriceTransformer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = lookback // patch_size
        
        # SOTA: RevIN for distribution shift
        self.revin = RevIN(num_features=input_dim)
        
        # SOTA: Patch Embedding (local temporal semantics)
        self.patch_embedding = nn.Linear(input_dim * patch_size, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder for target (t+1)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1) # Predicting Close price
        )

    def forward(self, x):
        # 1. Normalize (RevIN)
        x = self.revin(x, 'norm') # [Batch, Seq, Features]
        
        # 2. Patching
        # Reshape to [Batch, Num_Patches, Patch_Size * Features]
        batch_size = x.size(0)
        x = x.unfold(1, self.patch_size, self.patch_size) # [Batch, Num_Patches, Features, Patch_Size]
        x = x.transpose(2, 3).reshape(batch_size, self.num_patches, -1)
        
        # 3. Embedding + Transformer
        x = self.patch_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # 4. Decode
        out = self.decoder(x[:, -1, :]) # Use the last patch representation
        
        # 5. Denormalize target (using RevIN stats from the 'Close' feature)
        # Note: Denorm expects same shape as original input or specific stats.
        # We simplify here by manually denormalizing the target Close price.
        # Close is index 3 in OHLCV
        mean_close = self.revin.mean[:, :, 3]
        std_close = self.revin.stdev[:, :, 3]
        out = out * std_close + mean_close
        
        return out

# --- Data Handling ---
class StockDataset(Dataset):
    def __init__(self, ticker, lookback=64, period="2y", interval="1d"):
        # lookback must be divisible by patch_size (8)
        self.lookback = lookback
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        
        self.data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.lookback]
        y = self.data[idx+self.lookback, 3] # Close price
        return torch.tensor(x), torch.tensor(y).view(-1)

def train_model(ticker="BTC-USD", lookback=64, epochs=50):
    dataset = StockDataset(ticker, lookback=lookback)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = ResearchPriceTransformer(input_dim=5, lookback=lookback)
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    print(f"Training SOTA Research Transformer on {ticker}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(loader):.6f}")
    
    torch.save(model.state_dict(), "research_transformer.pth")
    print("Model saved to research_transformer.pth")

if __name__ == "__main__":
    train_model()
