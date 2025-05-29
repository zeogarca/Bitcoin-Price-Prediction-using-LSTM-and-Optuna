
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import fastai.tabular.all as ft
from torch.utils.data import Dataset
from fastai.callback.tracker import EarlyStoppingCallback
import optuna
from sklearn.preprocessing import MinMaxScaler

# Cargar datos
df = pd.read_csv("/content/btcusd_1-min_data.csv")
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.set_index('Date')[['Close']].dropna()

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(df[['Close']]).flatten()
T = torch.tensor(scaled_close, dtype=torch.float32)

class BTC_ds(Dataset):
    def __init__(self, t, l=30):
        self.t = t
        self.l = l

    def __len__(self):
        return len(self.t) - self.l - 1

    def __getitem__(self, i):
        return self.t[i:i+self.l].unsqueeze(1), self.t[i+self.l]

# Modelo LSTM
class BitcoinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Función objetivo para Optuna
def objective(trial):
    try:
        seq_len = trial.suggest_int('seq_len', 20, 60)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.0, 0.4)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        # train_ds = BTC_ds
        # valid_ds = BTC_ds

        print(f"Trial params: seq_len={seq_len}, hidden_size={hidden_size}, dropout={dropout:.3f}, lr={lr:.5f}, batch_size={batch_size}")
        print(f"Train size: {len(train_ds)}, Valid size: {len(valid_ds)}")

        if len(train_ds) < batch_size or len(valid_ds) < batch_size:
            print("Dataset demasiado pequeño para el batch size, descartando trial.")
            return float('inf')

        dls = ft.DataLoaders.from_dsets(train_ds, valid_ds, bs=batch_size, device='cuda:0')

        model = BitcoinLSTM(input_size=1, hidden_size=hidden_size, num_layers=2, dropout=dropout)

        learn = ft.Learner(
            dls,
            model,
            loss_func=ft.MSELossFlat(),
            opt_func=ft.ranger,
            cbs=EarlyStoppingCallback(monitor='valid_loss', patience=3)
        )

        learn.fit_one_cycle(1, lr)
        val_loss = learn.validate()[0]
        print(f"Validation loss: {val_loss:.6f}")
        return val_loss

    except Exception as e:
        print(f"Trial failed due to exception: {e}")
        return float('inf')

# Ejecutar Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2)

print("✅ Mejores hiperparámetros encontrados:")
print(study.best_params)

import joblib

# Guardar el estudio completo en un archivo
joblib.dump(study, "optuna_study_btc_lstm.pkl")

import matplotlib.pyplot as plt

losses = [t.value for t in study.trials if t.value is not None and np.isfinite(t.value)]

plt.figure(figsize=(10, 5))
plt.plot(losses, marker='o')
plt.title("Pérdida de validación por trial")
plt.xlabel("Trial")
plt.ylabel("Validación MSE")
plt.grid(True)
plt.show()

from optuna.visualization.matplotlib import plot_param_importances

plot_param_importances(study)
plt.show()

from optuna.visualization.matplotlib import plot_parallel_coordinate

plot_parallel_coordinate(study)
plt.show()

from optuna.visualization.matplotlib import plot_optimization_history

plot_optimization_history(study)
plt.show()

# Entrenar con los mejores parámetros
best_params = study.best_params
train_ds = BTC_ds(T[:-1000], best_params['seq_len'])
valid_ds = BTC_ds(T[-1000 - best_params['seq_len']:], best_params['seq_len'])
dls = ft.DataLoaders.from_dsets(train_ds, valid_ds, bs=best_params['batch_size'], device='cuda:0')

model = BitcoinLSTM(1, best_params['hidden_size'], 2, best_params['dropout'])
learn = ft.Learner(dls, model, loss_func=ft.MSELossFlat(), opt_func=ft.ranger)
learn.fit_one_cycle(1, best_params['lr'])

preds, targs = learn.get_preds()
preds = preds.cpu().numpy().flatten()
targs = targs.cpu().numpy().flatten()

preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
targs_inv = scaler.inverse_transform(targs.reshape(-1, 1)).flatten()

plt.figure(figsize=(12,6))
plt.plot(targs_inv, label="Real")
plt.plot(preds_inv, label="Predicción")
plt.title("Predicción vs Real - LSTM")
plt.xlabel("Tiempo")
plt.ylabel("Precio BTC (USD)")
plt.legend()
plt.grid(True)
plt.show()
