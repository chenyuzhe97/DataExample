import pandas as pd
import re


def parse_log_file(file_path):
    # 定义要提取的字段
    fields = ['Time', 'Data1', 'Data2', 'Data3', 'Data4',
              'Data5', 'Data6', 'Data7', 'Data8']

    # 初始化存储数据的列表
    data_list = []

    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 按分隔符分割条目
    entries = content.split('-----------------------')

    # 更新后的正则模式，用于提取字段和值，包括 Time 字段
    # 这个模式会匹配字段名和对应的值，无论值是否以 0x 开头
    pattern = re.compile(r'(\w+):\s*(0x[0-9a-fA-F]+|\d{2}:\d{2}:\d{2}\.\d{3})')

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue  # 跳过空条目

        # 初始化一个字典存储当前条目的数据
        entry_data = {field: None for field in fields}

        # 查找所有字段
        matches = pattern.findall(entry)
        print(f'匹配的字符串为：{matches}')
        for key, value in matches:
            if key == 'Time':
                entry_data['Time'] = value  # 保留为字符串或转换为时间戳
            elif key.startswith('Data'):
                try:
                    entry_data[key] = int(value, 16)  # 转换为 int32
                except ValueError:
                    entry_data[key] = None  # 如果转换失败，设置为 None

        # 检查是否至少包含 Time 和 Data1
        if entry_data['Time'] and entry_data['Data1'] is not None:
            data_list.append(entry_data)
        else:
            print(f"跳过不完整的条目:\n{entry}\n")

    # 创建 DataFrame
    df = pd.DataFrame(data_list)

    # 如果需要将 'Time' 转换为时间戳，可以使用以下代码
    # 这里假设日期信息可以从文件名中提取，例如 'output_20250102_201426_344.txt'
    # 提取日期部分 '20250102'
    import os
    filename = os.path.basename(file_path)
    date_match = re.search(r'output_(\d{8})_\d+_\d+\.txt', filename)
    if date_match:
        date_str = date_match.group(1)  # '20250102'
        # 将 'Time' 列与日期结合，转换为完整的 datetime 对象
        df['Timestamp'] = pd.to_datetime(
            date_str + ' ' + df['Time'],
            format='%Y%m%d %H:%M:%S.%f'
        )
        # 可以选择将 'Time' 列删除或保留
        # 这里保留 'Timestamp' 列，并将 'Time' 列删除
        df.drop(columns=['Time'], inplace=True)
        # 将 'Timestamp' 列设置为索引（可选）
        # df.set_index('Timestamp', inplace=True)
    else:
        print("无法从文件名中提取日期信息。")

    return df


# 使用示例
if __name__ == "__main__":
    log_file = './data/data_pet/data/output_20250102_201426_344.txt'  # 替换为您的文件路径
    df = parse_log_file(log_file)

    # 显示前几行数据
    print(df.head())

    # 保存为 CSV
    df.to_csv('output.csv', index=False)
    print("数据已成功保存为 output.csv")
