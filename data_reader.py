import os
import numpy as np
import pandas as pd
import re

class DataReaderProcessor:
    def __init__(self, mode='wifi', custom_step=None, save_dir='./processed_data'):
        """
        初始化数据处理类，设置模式和步长。

        :param mode: 数据处理模式 ('bluetooth', 'serial', 'wifi')
        :param custom_step: Wi-Fi模式下的自定义step值，默认为None
        :param save_dir: 处理后数据保存的目录
        """
        self.save_dir = save_dir  # 数据保存的目录

        # 设置默认的 step
        if mode == 'bluetooth':
            self.step = 0.002  # 蓝牙模式的步长
        elif mode == 'serial':
            self.step = 0.001  # 串口模式的步长
        elif mode == 'wifi' and custom_step is not None:
            self.step = custom_step  # Wi-Fi模式的自定义步长
        else:
            raise ValueError("Invalid mode or custom step not provided for Wi-Fi mode.")

    def parse_log_file(self, file_path):
        """
        解析txt日志文件，提取字段并转化为DataFrame。

        :param file_path: 待解析的txt文件路径
        :return: 返回一个包含时间和数据的DataFrame
        """
        # 定义要提取的字段
        fields = ['Time', 'Data1', 'Data2', 'Data3', 'Data4', 'Data5', 'Data6', 'Data7', 'Data8']

        # 初始化存储数据的列表
        data_list = []

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 按分隔符分割条目
        entries = content.split('-----------------------')

        # 更新后的正则模式，用于提取字段和值
        pattern = re.compile(r'(\w+):\s*(0x[0-9a-fA-F]+|\d{2}:\d{2}:\d{2}\.\d{3})')

        # 遍历每个条目
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue  # 跳过空条目

            # 初始化一个字典存储当前条目的数据
            entry_data = {field: None for field in fields}

            # 查找所有字段
            matches = pattern.findall(entry)
            for key, value in matches:
                if key == 'Time':
                    entry_data['Time'] = value  # 保留为字符串或转换为时间戳
                elif key.startswith('Data'):
                    try:
                        entry_data[key] = int(value, 16)  # 转换为 int32
                    except ValueError:
                        entry_data[key] = None  # 如果转换失败，设置为 None

            # 检查是否至少包含 Time 和 Data1...Data8
            if entry_data['Time'] is not None and entry_data['Data1'] is not None:
                data_list.append(entry_data)

        # 将提取的数据转换为DataFrame
        df = pd.DataFrame(data_list, columns=fields)
        return df

    def process_file(self, file_path):
        """
        处理数据文件，生成按列保存的结果文件。

        :param file_path: 待处理数据的文件路径
        """
        # 判断文件后缀名用于使用对应方法
        _, ext = os.path.splitext(file_path)

        # 读取数据时跳过第一行，并且读取第1列到第8列的数据
        if ext.lower() == '.csv':
            data = pd.read_csv(file_path, header=None, skiprows=2, delimiter=',', low_memory=False)
            # 获取数据的列，从第1列到第8列（原始数据）
            columns = data.iloc[:, 0:8].values  # 从第1列到第8列
        elif ext.lower() == '.txt':
            data = self.parse_log_file(file_path)
            # 获取数据的列，从第1列到第8列（原始数据）
            columns = data.iloc[:, 1:9].values  # 从第1列到第8列
        else:
            raise ValueError(f"Unsupported file extension: {ext}")


        # 生成时间序列，时间从0开始，每次间隔为step
        time = np.arange(0, len(columns) * self.step, self.step)

        # 创建保存文件的文件夹（如果没有的话）
        for i in range(1, 9):
            folder_path = os.path.join(self.save_dir, str(i))  # 生成每个文件夹的路径
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        # 获取原始文件的名字（去掉扩展名）
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        # 处理每一列数据
        for col_idx in range(columns.shape[1]):
            # 获取当前列的数据
            value = columns[:, col_idx]

            # 将时间和当前列的值合并
            result = pd.DataFrame({'time': time, 'value': value})

            # 保存到指定目录中的对应文件夹里
            result.to_csv(os.path.join(self.save_dir, str(col_idx + 1), f"{base_filename}_col{col_idx + 1}.csv"),
                          index=False)

            print(f"Data for column {col_idx + 1} saved to folder {col_idx + 1}")


if __name__ == '__main__':
    # 示例：创建 DataProcessor 实例并调用处理方法
    processor_bluetooth = DataReaderProcessor(mode='serial', save_dir="data/data_separation/")
    processor_bluetooth.process_file("./data/data_pet/data/output_20250102_201426_344.txt")

    # processor_serial = DataProcessor(mode='serial', save_dir="./data/data_separation")
    # processor_serial.process_file("./data/data_origin/WAVE(2025.1.6-16.09.35).csv")
    #
    # # Wi-Fi模式，step由用户自定义
    # processor_wifi = DataProcessor(mode='wifi', custom_step=0.005, save_dir="./data/data_separation")
    # processor_wifi.process_file("./data/data_origin/WAVE(2025.1.6-16.09.35).csv")
