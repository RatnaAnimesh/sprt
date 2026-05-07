import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
import os

# --- Kalman Filter (from train_nn.py) ---
class KalmanFilter:
    def __init__(self, Q=0.01, R=0.25):
        self.Q, self.R = Q, R
        self.x, self.P = None, 1.0

    def update(self, z):
        if self.x is None: self.x = z; return self.x, self.P
        self.P += self.Q
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x, self.P

# --- Transformer Model Architecture ---
class FinancialTransformer(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=4, num_layers=2, dropout=0.1):
        super(FinancialTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model)) # Max sequence length of 100
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, 1) # Predict normalized mid-price change

    def forward(self, x):
        # x: [Batch, SeqLen, InputDim]
        x = self.embedding(x) # [Batch, SeqLen, d_model]
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        output = self.transformer_encoder(x)
        # Take the final step representation for the prediction
        last_step = output[:, -1, :]
        return self.decoder(last_step)

# --- Data Loading (Sequential for Transformer) ---
def load_sequential_data(files, lookback=30):
    X, y = [], []
    for file in files:
        rows = []
        with open(file, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                if row['product'] == 'TOMATOES':
                    rows.append(row)
        
        kf = KalmanFilter()
        smooth_prices = []
        for r in rows:
            sx, _ = kf.update(float(r['mid_price']))
            smooth_prices.append(sx)
            
        feature_matrix = []
        for i in range(len(rows)):
            row = rows[i]
            mid = smooth_prices[i]
            bp1, bv1 = float(row['bid_price_1']), float(row['bid_volume_1'])
            ap1, av1 = float(row['ask_price_1']), float(row['ask_volume_1'])
            
            # Feature normalization
            obi = (bv1 - av1) / (bv1 + av1) if (bv1 + av1) > 0 else 0.0
            velocity = (mid - smooth_prices[i-1]) if i > 0 else 0.0
            spread = ap1 - bp1
            
            # [normalized_price, velocity, volume_imb, spread]
            feat = [
                (mid - 5000.0) / 10.0,
                velocity,
                obi,
                spread / 10.0,
                bv1 / 100.0,
                av1 / 100.0
            ]
            feature_matrix.append(feat)

        # Create overlapping windows
        for i in range(lookback, len(feature_matrix) - 1):
            window = feature_matrix[i-lookback:i]
            target = feature_matrix[i+1][0] # Goal: Predict next normalized price
            X.append(window)
            y.append(target)
            
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)

if __name__ == "__main__":
    lookback = 30
    print(f"--- Temporal Transformer Implementation ---")
    data_files = [
        "/Users/ashishmishra/imc-prosperity/TUTORIAL_ROUND_1/prices_round_0_day_-1.csv",
        "/Users/ashishmishra/imc-prosperity/TUTORIAL_ROUND_1/prices_round_0_day_-2.csv"
    ]
    
    if not all(os.path.exists(f) for f in data_files):
        print("Error: CSV files not found. Ensure paths are correct.")
    else:
        X, y = load_sequential_data(data_files, lookback=lookback)
        print(f"Loaded {len(X)} sequences of length {lookback}.")

        # Split data (simpler than scikit-learn to keep dependencies low)
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Initialize Model
        model = FinancialTransformer(input_dim=6)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training Loop
        epochs = 10
        batch_size = 64
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(X_train.size()[0])
            epoch_loss = 0
            
            for i in range(0, X_train.size()[0], batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(X_test)
                val_loss = criterion(val_preds, y_test)
                
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss/(len(X_train)/batch_size):.6f}, Val Loss: {val_loss.item():.6f}")

        print("\n--- Model Training Complete ---")
        # Save placeholder for future integration
        torch.save(model.state_dict(), "transformer_model.pth")
        print("Model weights saved to transformer_model.pth")
