import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def periods_where_pv_is_null(df_inputs: pd.DataFrame):
    nb_days = int(df_inputs[df_inputs['ZONE_1'] == 0]['POWER'].shape[0] / 144)
    max_zone1 = df_inputs[df_inputs['ZONE_1'] == 0]['POWER'].values.reshape(nb_days, 144).max(axis=0)
    indices1 = np.where(max_zone1 == 0)[0]
    return indices1

def build_pv_features(df_var: pd.DataFrame, indices: np.array):
    n_days = int(len(df_var) / 144)
    y = df_var['POWER'].values.reshape(n_days, 144)
    y = np.delete(y, indices, axis=1)
    wrh = df_var['Weather_Relative_Humidity'].values.reshape(n_days, 144)
    wrh = np.delete(wrh, indices, axis=1)
    ghr = df_var['Global_Horizontal_Radiation'].values.reshape(n_days, 144)
    ghr = np.delete(ghr, indices, axis=1)
    dhr = df_var['Diffuse_Horizontal_Radiation'].values.reshape(n_days, 144)
    dhr = np.delete(dhr, indices, axis=1)
    wdr = df_var['Weather_Daily_Rainfall'].values.reshape(n_days, 144)
    wdr = np.delete(wdr, indices, axis=1)
    TT = df_var['Weather_Temperature_Celsius'].values.reshape(n_days, 144)
    TT = np.delete(TT, indices, axis=1)
    x = np.concatenate([TT, ghr, dhr, wdr, wrh], axis=1)
    return x, y

def pv_data(path_name: str, test_size: int, random_state: int = 0):
    df_pv = pd.read_csv(path_name, parse_dates=True, index_col=0)
    indices = periods_where_pv_is_null(df_inputs=df_pv)
    d_index = df_pv['POWER'].asfreq('D').index
    x, y = build_pv_features(df_var=df_pv, indices=indices)
    df_y = pd.DataFrame(data=y, index=d_index)
    df_x = pd.DataFrame(data=x, index=d_index)
    df_x_train, df_x_TEST, df_y_train, df_y_TEST = train_test_split(df_x, df_y, test_size=test_size,
                                                                    random_state=random_state, shuffle=True)
    df_x_LS, df_x_VS, df_y_LS, df_y_VS = train_test_split(df_x_train, df_y_train, test_size=test_size,
                                                          random_state=random_state, shuffle=True)
    return [df_x_LS, df_y_LS, df_x_VS, df_y_VS, df_x_TEST, df_y_TEST], indices

def scale_data_multi(x_LS: np.array, y_LS: np.array, x_VS: np.array, y_VS: np.array, x_TEST: np.array,
                     y_TEST: np.array):
    y_LS_scaler = StandardScaler()
    y_LS_scaler.fit(y_LS)
    y_LS_scaled = y_LS_scaler.transform(y_LS)
    y_VS_scaled = y_LS_scaler.transform(y_VS)
    y_TEST_scaled = y_LS_scaler.transform(y_TEST)
    x_LS_scaler = StandardScaler()
    x_LS_scaler.fit(x_LS)
    x_LS_scaled = x_LS_scaler.transform(x_LS)
    x_VS_scaled = x_LS_scaler.transform(x_VS)
    x_TEST_scaled = x_LS_scaler.transform(x_TEST)
    return x_LS_scaled, y_LS_scaled, x_VS_scaled, y_VS_scaled, x_TEST_scaled, y_TEST_scaled, y_LS_scaler

class LoadDataset(Dataset):
    def __init__(self, x, loads, scaler):
        self.x = x
        self.loads = loads
        self.y_scaler = scaler

    def __len__(self):
        return self.loads.shape[0]

    def __getitem__(self, idx):
        return self.loads[idx], self.x[idx]

def get_datasets(path_name: str, test_size: int = 20, random_state: int = 0):
    data, indices = pv_data(path_name, test_size, random_state)
    df_x_LS, df_y_LS = data[0], data[1]
    df_x_VS, df_y_VS = data[2], data[3]
    df_x_TEST, df_y_TEST = data[4], data[5]
    x_LS_scaled, y_LS_scaled, x_VS_scaled, y_VS_scaled, x_TEST_scaled, y_TEST_scaled, y_LS_scaler = scale_data_multi(
        x_LS=df_x_LS.values, y_LS=df_y_LS.values, x_VS=df_x_VS.values, y_VS=df_y_VS.values, x_TEST=df_x_TEST.values,
        y_TEST=df_y_TEST.values)
    train_dataset = LoadDataset(x_LS_scaled, y_LS_scaled, y_LS_scaler)
    val_dataset = LoadDataset(x_VS_scaled, y_VS_scaled, y_LS_scaler)
    test_dataset = LoadDataset(x_TEST_scaled, y_TEST_scaled, y_LS_scaler)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return df_y_LS, df_x_LS, y_LS_scaler, train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader


