import math
from math import sqrt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataload import get_datasets
from model import EpsilonThetaCond, T, alpha_bar_t, noise_schedule, beta_tild_t, alpha_t
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import datetime
from joblib import dump, load
import timeit
from scipy.stats import gaussian_kde

def save_data_with_pickle(data, filename, folder='processed_data'):
    """将数据保存到Pickle文件中."""
    file_path = os.path.join(folder, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
def time_format(t):
    if t > 60 * 60:
        h = t // (60 * 60)
        m = t % (60 * 60)
        s = m % 60
        m = m // 60
        return f"{h} hours {m} minutes {s} seconds"
    elif t > 60:
        s = t % (60)
        m = t // 60
        return f"{m} minutes {s} seconds"
    else:
        return f"{t} seconds"

# 调用get_datasets函数
path_name = 'solar_1516.csv'
test_size = 20
random_state = 0
tag = 'pv'
df_y_LS, df_x_LS, y_LS_scaler, train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = get_datasets(path_name, test_size, random_state)

save_data_with_pickle(df_y_LS, 'df_y_LS.pkl')
save_data_with_pickle(df_x_LS, 'df_x_LS.pkl')
save_data_with_pickle(y_LS_scaler, 'y_LS_scaler.pkl')
save_data_with_pickle(train_dataset, 'train_dataset.pkl')
save_data_with_pickle(val_dataset, 'val_dataset.pkl')
save_data_with_pickle(test_dataset, 'test_dataset.pkl')
save_data_with_pickle(train_dataloader, 'train_dataloader.pkl')
save_data_with_pickle(val_dataloader, 'val_dataloader.pkl')
save_data_with_pickle(test_dataloader, 'test_dataloader.pkl')

print(df_y_LS.values.shape)  ##df_y_LS (996,66)
print(df_x_LS.values.shape)  ##df_x_LS (996,600=66*6)

print("Scaler:", y_LS_scaler)
print("Train Dataset Length:", len(train_dataset))
print("Validation Dataset Length:", len(val_dataset))
print("Test Dataset Length:", len(test_dataset))

# Training parameters
n_epoch = 20000
train_input = train_dataloader
test_input = test_dataloader
losses = []
validation_losses = []
tmp_losses = []

early_stop_counter = 0
early_stop_threshold = 0.02
early_stop_max_epochs = 10
stop_training = False
cond_l = train_dataset[0][1].shape[0]
target_d = train_dataset[0][0].shape[0]
print(f"cond length : {cond_l}")
print(f"target dim: {target_d}")

residual_layers = 8
residual_channels = 8
dil = 2

type_model = "conditional"
model = EpsilonThetaCond(cond_length=cond_l, target_dim=target_d, residual_layers=residual_layers,
                         residual_channels=residual_channels, dilation_cycle_length=dil)
model.train()
criterion = nn.MSELoss()

learning_rate = 0.0001  # 设置初始学习率
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
shuffle = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Model of type : {type_model}")
num_params = sum(param.numel() for param in model.parameters())
print(f"Number of parameters of the model : {num_params}")
print(device)
print("T: ", T)

# Training
start = timeit.default_timer()
for i in range(n_epoch):

    if i % 100 == 0:
        print(f"Starting Epoch {i + 1} / {n_epoch}")
    #   Training loop
    tmp_losses = []
    model.train()
    for batch in train_input:  # load is a batch
        load = batch[0]
        x = batch[1]
        iterate = [int(np.random.rand() * T)]
        for t in iterate:
            optimizer.zero_grad()
            noise = torch.randn(load.shape)
            noisy_input = sqrt(alpha_bar_t[t]) * load + sqrt(1 - alpha_bar_t[t]) * noise  # [Batch_size, 24]
            noisy_input = noisy_input.unsqueeze(1)  # [Batch_size, 1 ,24] (only if playng with convs)
            # Deal with gpu training
            noisy_input = noisy_input.to(device)
            noise = noise.to(device)
            x = x.to(device)
            t = torch.tensor(t)
            t = t.to(device)
            # Forward pass and loss computation
            out = model(noisy_input.float(), t, x.unsqueeze(1).float())
            out = out.squeeze(1)  # Only if playing with convolutions

            kappa = (noise_schedule[t] ** 2) / (2 * beta_tild_t[t] * alpha_t[t] * (1 - alpha_bar_t[t]))
            loss = kappa * criterion(out, noise)
            # loss = criterion(out, noise)
            loss.backward()
            tmp_losses.append(loss.item())
            optimizer.step()  # When to update weights ? -> each noise step or each sample ?
    losses.append(np.mean(tmp_losses))
    stop = timeit.default_timer()

name_model = 'models/exp_T/' + tag + '_' + type_model + '_' + 'model_diff_' + str(
    T) + '_epoch_' + str(n_epoch)
name_model += '.pth'
torch.save(model.state_dict(), name_model)

plt.plot(losses)
plt.title("Training loss")
plt.show()
plt.clf()