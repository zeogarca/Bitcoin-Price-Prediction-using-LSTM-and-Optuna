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

def objective(trial):
    seq_len = trial.suggest_int('seq_len', 20, 60)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.0, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    train_ds = BTC_ds(T[:-1000], seq_len)
    valid_ds = BTC_ds(T[-1000 - seq_len:], seq_len)
    dls = ft.DataLoaders.from_dsets(train_ds, valid_ds, bs=batch_size, device='cuda:0')

    model = BitcoinLSTM(input_size=1, hidden_size=hidden_size, num_layers=2, dropout=dropout)

    learn = ft.Learner(
        dls,
        model,
        loss_func=ft.MSELossFlat(),
        opt_func=ft.ranger,
        cbs=EarlyStoppingCallback(monitor='valid_loss', patience=3)
    )

    learn.fit_one_cycle(10, lr)
    return learn.validate()[0]

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

print("Mejores hiperparámetros encontrados:")
print(study.best_params)