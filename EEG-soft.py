import pandas as pd
import numpy as np
import mne
from mne.io import RawArray

import os

# 设置要遍历的文件夹路径
folder_path = './data/data_separation'

# 使用os.walk遍历文件夹及其子文件夹
file_names = []

for root, dirs, files in os.walk(folder_path):
    for file in files:
        # 获取相对路径（从folder_path开始）
        relative_path = os.path.relpath(os.path.join(root, file), folder_path)
        # 将文件的相对路径添加到列表
        file_info.append({'FilePath': relative_path})


# # 读取CSV文件
# file_path = './data/data_separation/2/WAVE(2025-01-15-20.02.12)_col2.csv'
# data = pd.read_csv(file_path)
#
# # 提取时间和信号
# times = data['time'].values  # 获取时间列
# values = data['value'].values  # 获取信号强度列
#
# # 设定采样率（假设时间是等间隔的）
# sfreq = 1 / (times[1] - times[0])  # 采样率（基于时间差计算）
#
# # 创建通道信息（假设只有一个EEG通道）
# info = mne.create_info(ch_names=['EEG'], sfreq=sfreq, ch_types=['eeg'])
#
# # 将信号转换为Raw对象
# raw = RawArray(values[np.newaxis, :], info)  # 数据需要是二维的（通道数，样本数）
#
# # 绘制EEG信号
# raw.plot(duration=10, n_channels=1)
