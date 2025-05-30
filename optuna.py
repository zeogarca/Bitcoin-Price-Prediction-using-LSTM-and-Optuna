# -*- coding: utf-8 -*-
"""optuna

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BW-tTtKsiSK8NDJ8FNiswvFcHVzIi9IQ
"""

!pip install optuna

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import optuna
import matplotlib.pyplot as plt

# Cargar dataset
df = pd.read_csv("/content/btcusd_1-min_data.csv")

# Preprocesamiento
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.set_index('Date')[['Close']].copy()
df.dropna(inplace=True)

# Escalado
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[['Close']].values).flatten()

# Función para crear secuencias
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# Modelo LSTM
class BitcoinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(BitcoinLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def objective(trial):
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Trial {trial.number} usando dispositivo: {device}")

    try:
        # Hiperparámetros sugeridos por Optuna
        params = {
            'seq_len': trial.suggest_int('seq_len', 10, 50),
            'hidden_size': trial.suggest_categorical('hidden_size', [16, 32, 64, 128]),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
        }

        # Crear secuencias
        X, y_seq = create_sequences(scaled_close, params['seq_len'])
        print(f"📊 Trial {trial.number}: {len(X)} secuencias creadas con seq_len={params['seq_len']}")

        if len(X) < 500:
            print(f"⚠️ Trial {trial.number}: muy pocos datos, saltando.")
            return float('inf')

        # División temporal
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

        # Tensores de entrenamiento
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=params['batch_size'], shuffle=True)

        # Validación por batches también (¡importante!)
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=params['batch_size'], shuffle=False)

        # Modelo
        model = BitcoinLSTM(
            input_size=1,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.MSELoss()

        # Entrenamiento
        best_val_loss = float('inf')
        for epoch in range(20):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validación por batch (¡esto evita el OOM!)
            model.eval()
            val_loss_total = 0.0
            count = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_preds = model(X_batch)
                    loss = criterion(val_preds, y_batch).item()
                    val_loss_total += loss * X_batch.size(0)
                    count += X_batch.size(0)
            val_loss = val_loss_total / count

            # Reporte y poda
            trial.report(val_loss, epoch)
            if trial.should_prune():
                print(f"🪓 Trial {trial.number} podado en epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        print(f"✅ Trial {trial.number} terminado con val_loss = {best_val_loss:.5f}")
        return best_val_loss

    except Exception as e:
        print(f"❌ Error en trial {trial.number}: {e}")
        return float('inf')

import optuna
optuna.logging.set_verbosity(optuna.logging.DEBUG)

# Optimización con Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1, show_progress_bar=True)

print("🔍 Mejores hiperparámetros:")
print(study.best_params)

# Optimización con Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5, show_progress_bar=True)

print("🔍 Mejores hiperparámetros:")
print(study.best_params)

# Entrenamiento final con mejores parámetros
best_params = study.best_params
X, y_seq = create_sequences(scaled_close, best_params['seq_len'])
# División en train, val y test
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y_seq[:train_size], y_seq[train_size:train_size+val_size], y_seq[train_size+val_size:]


# DataLoader de entrenamiento
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    ),
    batch_size=best_params['batch_size'], shuffle=True
)

# DataLoader de validación
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    ),
    batch_size=best_params['batch_size'], shuffle=False
)

model = BitcoinLSTM(
    input_size=1,
    hidden_size=128,
    num_layers=1,
    dropout=0.12592806791831518
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00011195860024489018)
criterion = nn.MSELoss()

# Entrenamiento completo
for epoch in range(15):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Evaluación final en el set de prueba
model.eval()
with torch.no_grad():
    test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    preds = model(test_tensor).numpy()

# Desescalar
actual = scaler.inverse_transform(y_test.reshape(-1, 1))
predicted = scaler.inverse_transform(preds)

# Visualización
plt.figure(figsize=(12, 5))
plt.plot(actual, label='Real')
plt.plot(predicted, label='Predicción')
plt.legend()
plt.title("Predicción del precio de Bitcoin (Optuna)")
plt.show()