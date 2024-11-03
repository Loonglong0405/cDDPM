from math import sqrt
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from dataload import get_datasets
from model import EpsilonThetaCond, T, alpha_bar_t, noise_schedule, beta_tild_t, alpha_t
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load
import timeit
from scipy.stats import gaussian_kde
def load_data_from_pickle(filename, folder='processed_data'):
    """从Pickle文件中加载数据."""
    file_path = os.path.join(folder, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

y_LS_scaler = load_data_from_pickle('y_LS_scaler.pkl')
test_dataset = load_data_from_pickle('test_dataset.pkl')
train_dataset = load_data_from_pickle('train_dataset.pkl')
val_dataset = load_data_from_pickle('val_dataset.pkl')

def sample_infer_cond(model, n_s=1, x_cond=None):
    x_t = torch.randn(n_s, target_d)
    context = torch.tensor(np.tile(x_cond, n_s).reshape(n_s, cond_l)).float()

    for t in range(T - 1, -1, -1):
        z = torch.randn(x_t.shape) if t > 1 else 0
        t = torch.tensor(t)
        t = t.to(device)

        eps_theta_t = model(x_t.unsqueeze(1).to(device), t, context.unsqueeze(1).to(device))
        eps_theta_t = eps_theta_t.squeeze(1)

        mu = (x_t - ((noise_schedule[t] / sqrt(1 - alpha_bar_t[t])) * eps_theta_t.to('cpu'))) / sqrt(alpha_t[t])
        sigma = torch.ones(mu.shape) * sqrt(beta_tild_t[t])

        x_t = mu + sigma * z

    scenarios = x_t

    return scenarios

def buildScenarioscond(model):
    n_s = 100
    n_days = 20
    scenarios = []
    for i in range(n_days):
        sample = sample_infer_cond(model, n_s=n_s, x_cond=val_dataset[i][1])
        sample = sample.detach().numpy()
        sample = y_LS_scaler.inverse_transform(sample)
        scenarios.append(sample)

    return np.transpose(np.array(scenarios), (0, 2, 1)), test_dataset  # (n_days, 24, n_s)

n_epoch=8000
residual_layers = 8
residual_channels = 8
cond_l = train_dataset[0][1].shape[0]
target_d = train_dataset[0][0].shape[0]
print(f"cond length : {cond_l}")
print(f"target dim: {target_d}")
dil = 2
tag= 'pv'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = EpsilonThetaCond(cond_length=cond_l, target_dim=target_d, residual_layers=residual_layers,
                         residual_channels=residual_channels, dilation_cycle_length=dil)
model.load_state_dict(torch.load(f'models/exp_T/pv_conditional_model_diff_350_epoch_8000.pth',map_location=torch.device('cpu')))

model.to(device)

scenarios, ground_truth = buildScenarioscond(model)
print(scenarios.shape)

with open('scenarios/exp_T/' + tag + "_diff_" + str(T) + "_epoch_" + str(n_epoch) + "_shuffle" + ".npy", 'wb') as f:
    np.save(f, scenarios)

with open('scenarios/exp_T/' + tag + "_gt" + "_diff_" + str(T) + ".npy", 'wb') as f:
    np.save(f, y_LS_scaler.inverse_transform(train_dataset[0:len(train_dataset)][0]))