# fine_tune_insightface.py
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from prometheus_client import Gauge, start_http_server

import psutil


device = "cuda" if torch.cuda.is_available() else "cpu"



EPOCH_GAUGE = Gauge("model_epoch", "Current epoch of the model")
LOSS_GAUGE = Gauge("model_loss", "Current loss of the model")
CPU_USAGE_GAUGE = Gauge("process_cpu_percent", "CPU usage percentage")
RAM_USAGE_GAUGE = Gauge("process_ram_bytes", "RAM usage in bytes")
# --- Dataset from CSV ---
class EmbeddingDataset(Dataset):
    def __init__(self, csv_file, id_map=None):
        self.df = pd.read_csv(csv_file)
        if "id" not in self.df.columns:
            raise ValueError("'id' column missing in CSV")
        
        # Extract features (all columns except 'id')
        self.X = self.df.drop(columns=["id"]).values.astype(np.float32)

        # Map IDs to 0..num_classes-1
        if id_map is None:
            unique_ids = sorted(self.df["id"].unique())
            self.id_map = {orig_id: idx for idx, orig_id in enumerate(unique_ids)}
        else:
            self.id_map = id_map

        # Map target IDs to integers
        self.y = self.df["id"].map(self.id_map).values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), self.y[idx]

# --- Simple MLP to Fine-Tune Embeddings ---
# fine_tune_insightface.py
class ArcFaceFineTuner(nn.Module):
    def __init__(self, input_dim, embedding_dim=512, num_classes=None):
        super().__init__()
        # Ensure input_dim matches the size of embeddings you pass
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        emb = self.fc(x)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        if self.training:
            return self.classifier(emb)
        return emb


# --- Load CSVs and create ID mapping ---
train_csv = "train.csv"
test_csv = "test.csv"

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

if "id" not in train_df.columns or "id" not in test_df.columns:
    raise ValueError("'id' column missing in CSV files")

# Determine embedding dimension dynamically
input_dim = train_df.shape[1] - 1  # exclude 'id' column
unique_ids = sorted(train_df["id"].unique())
id_map = {orig_id: idx for idx, orig_id in enumerate(unique_ids)}
num_classes = len(unique_ids)

# Create datasets and loaders
train_dataset = EmbeddingDataset(train_csv, id_map=id_map)
test_dataset = EmbeddingDataset(test_csv, id_map=id_map)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Initialize model ---
model = ArcFaceFineTuner(input_dim=input_dim, embedding_dim=512, num_classes=num_classes).to(device)

# --- Loss & optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# --- Training loop ---
for epoch in range(10):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Dimension check
        if X_batch.shape[1] != model.fc.in_features:
            raise ValueError(f"Embedding dimension mismatch: {X_batch.shape[1]} vs {model.fc.in_features}")

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Update CPU/RAM per batch if you want more granularity
        CPU_USAGE_GAUGE.set(psutil.cpu_percent())
        RAM_USAGE_GAUGE.set(psutil.virtual_memory().used)

    # Update epoch and average loss
    avg_loss = total_loss / len(train_loader)
    EPOCH_GAUGE.set(epoch)
    LOSS_GAUGE.set(avg_loss)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, CPU: {psutil.cpu_percent()}%, RAM: {psutil.virtual_memory().used / 1e9:.2f} GB")
# --- Save fine-tuned model ---
torch.save(model.state_dict(), "insightface_finetuned.pt")
print("Fine-tuned model saved as 'insightface_finetuned.pt'")
