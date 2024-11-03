from math import sqrt
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from dataload import get_datasets
import matplotlib.pyplot as plt
import pickle


def load_data_from_pickle(filename, folder='processed_data'):
    """从Pickle文件中加载数据."""
    file_path = os.path.join(folder, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
tag='pv'
df_y_LS = load_data_from_pickle('df_y_LS.pkl')
y_LS_scaler = load_data_from_pickle('y_LS_scaler.pkl')
train_dataset = load_data_from_pickle('train_dataset.pkl')

with open("scenarios/exp_T/pv_diff_350_epoch_8000_shuffle.npy", 'rb') as f:
    s_TEST_400 = np.load(f)
with open("scenarios/exp_T/pv_gt_diff_400.npy", 'rb') as f:
    gt_400 = np.load(f)

def rebuild(null_zones, x):
    new_x = []
    for pv in x:
        new_pv = np.ones(144)
        new_pv[null_zones] = 0
        new_pv[np.where(new_pv == 1)] = pv
        new_x.append(new_pv)
    return np.array(new_x)

def rebuild_scenar(null_zones, x):
    new_x = []
    for pv_scenarios in x:
        new_pv = np.ones((pv_scenarios.shape[0] + len(null_zones), pv_scenarios.shape[1]))
        new_pv[null_zones, :] = 0
        prev_shape = new_pv.shape
        new_pv[np.where(new_pv == 1)] = pv_scenarios.reshape(new_pv[np.where(new_pv == 1)].shape)
        new_pv.reshape(prev_shape)
        new_x.append(new_pv)
    return np.array(new_x)

gt_400 = rebuild([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 , 26 ,  27 ,  28  , 29  , 30  , 31 ,  32 ,  33 ,  34 ,  35
  , 36  , 37  , 38  , 39  , 40  , 41 , 108 , 109 , 110 , 111 , 112 , 113 , 114 , 115,  116 , 117 , 118 , 119,
 120 , 121 , 122 , 123 , 124 , 125,  126 , 127 , 128 , 129 , 130 , 131 , 132 , 133 , 134 , 135 , 136 , 137
 , 138 , 139,  140 , 141 , 142 , 143], gt_400)

s_TEST_400 = rebuild_scenar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 , 26 ,  27 ,  28  , 29  , 30  , 31 ,  32 ,  33 ,  34 ,  35
  , 36  , 37  , 38  , 39  , 40  , 41 , 108 , 109 , 110 , 111 , 112 , 113 , 114 , 115,  116 , 117 , 118 , 119,
 120 , 121 , 122 , 123 , 124 , 125,  126 , 127 , 128 , 129 , 130 , 131 , 132 , 133 , 134 , 135 , 136 , 137
 , 138 , 139,  140 , 141 , 142 , 143], s_TEST_400)

s_TEST_400[s_TEST_400 < 0] = 0
point = 7  # 您指定的点

n_s = 100
N_q = 99

# 计算指定的分位数
percentiles = [0, 5, 10, 20, 50, 80, 90, 95, 100]
q_set = [p/100 for p in percentiles]
quantiles = np.quantile(s_TEST_400[point], q=q_set, axis=1)

# 创建DataFrame
df = pd.DataFrame({f'{p}%': quantiles[i] for i, p in enumerate(percentiles)})

# 添加ground truth
df['Ground Truth'] = gt_400[point]
# 添加横坐标 X（时间点）
df['Time'] = range(144)
# 重新排列列的顺序，使 'Time' 成为第一列
columns = ['Time'] + [col for col in df.columns if col != 'Time']
df = df[columns]

# 保存到Excel
output_dir = 'output_plots'
os.makedirs(output_dir, exist_ok=True)
excel_path = os.path.join(output_dir, f'quantiles_and_gt_day_{point}.xlsx')
df.to_excel(excel_path, index=False)

print(f"数据已保存到: {excel_path}")

# 计算覆盖率
def calculate_coverage(gt, lower, upper):
    return np.mean((gt >= lower) & (gt <= upper))

coverage_0_100 = calculate_coverage(df['Ground Truth'], df['0%'], df['100%'])
coverage_5_95 = calculate_coverage(df['Ground Truth'], df['5%'], df['95%'])
coverage_10_90 = calculate_coverage(df['Ground Truth'], df['10%'], df['90%'])
coverage_20_80 = calculate_coverage(df['Ground Truth'], df['20%'], df['80%'])
print(f"覆盖率 (0-100): {coverage_0_100:.2%}")
print(f"覆盖率 (5-95): {coverage_5_95:.2%}")
print(f"覆盖率 (10-90): {coverage_10_90:.2%}")
print(f"覆盖率 (20-80): {coverage_20_80:.2%}")
# 计算区间宽度
interval_width = np.mean(df['100%'] - df['0%'])
print(f"平均区间宽度 (0-100): {interval_width:.4f}")

# 绘图代码（与之前相同）
plt.figure(figsize=(9, 6))
plt.ylim(0, 10)
plt.xlabel("Time(h)")
plt.ylabel("Power(MW)")
plt.plot(s_TEST_400[point, :, :], color='gray', linewidth=1.5, alpha=0.6)
plt.plot(gt_400[point], color='red', label='Ground Truth')
plt.plot(quantiles[4], color='k', linewidth=2, label='50%')  # 50% 分位线

x_ticks = np.linspace(0, 144, 7)
x_labels = [0, 4, 8, 12, 16, 20, 24]
plt.xticks(x_ticks, x_labels)
y_ticks = np.arange(0, 11, 2)
plt.yticks(y_ticks)
plt.xlim(0, 144)
plt.ylim(0, 10)

for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

plt.tick_params(axis='both', which='both', width=2.0, direction='in', length=6)
plt.tick_params(axis='x', which='both', top=True, bottom=True, labelbottom=True)
plt.tick_params(axis='y', which='both', left=True, right=True, labelleft=True)

plt.grid(False)
plt.legend()
plt.savefig(os.path.join(output_dir, f'Power_{point}.png'), dpi=600, bbox_inches='tight')
plt.show()

def autocorrelation(data):
    n = len(data)
    mean_data = np.mean(data)
    data -= mean_data
    autocorr = np.correlate(data, data, mode='full')[n-1:]
    autocorr /= autocorr[0]  # normalize the autocorrelation by the zero lag
    return autocorr

auto_corr_gt = autocorrelation(gt_400[point])
auto_corr_scenarios = np.array([autocorrelation(s_TEST_400[point, :, i]) for i in range(n_s)])
mean_auto_corr_scenarios = np.mean(auto_corr_scenarios, axis=0)


os.makedirs(output_dir, exist_ok=True)

# 设置图形样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24

# 绘制自相关图
plt.figure(figsize=(9, 6))  # 3:2 比例
plt.plot(auto_corr_gt, color='blue', linewidth=2, label = 'Real')
plt.plot(mean_auto_corr_scenarios, color='red', linewidth=2, label = 'Generated')

# 设置坐标轴范围和刻度
x_ticks = [0, 24, 48, 72, 96, 120, 144]
y_ticks = np.arange(-1, 1.01, 0.5)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.xlim(0, 144)
plt.ylim(-1, 1)

# 设置边框线宽
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)

# 设置刻度
plt.tick_params(axis='both', which='both', width=2.0, direction='in', length=6)
plt.tick_params(axis='x', which='both', top=True, bottom=True, labelbottom=True)
plt.tick_params(axis='y', which='both', left=True, right=True, labelleft=True)

# 移除网格线
plt.grid(False)

legend = plt.legend(loc='upper right', fontsize=24, frameon=False)

# 添加轴标签
plt.xlabel('Lag(10min)')
plt.ylabel('Autocorrelation Coefficient')

# 保存图像
plt.savefig(os.path.join(output_dir, f'autocorrelation_day_{point}.png'), dpi=600, bbox_inches='tight')
plt.close()