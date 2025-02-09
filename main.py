import data_filter
import data_reader
import pandas as pd
import glob
import os
import re
import shutil


def copy_png_files_to_all_directory(root_directory):
    # 构建目标目录 './result/picture/all'
    all_directory = os.path.join(root_directory, 'all')

    # 如果目标目录不存在，创建它
    if not os.path.exists(all_directory):
        os.makedirs(all_directory)

    # 遍历 root_directory 目录及其所有子目录
    for subdir, _, files in os.walk(root_directory):
        # 排除 all 目录本身
        if os.path.basename(subdir) == 'all':
            continue

        for filename in files:
            # 检查文件是否为 png 文件
            if filename.endswith('.png'):
                file_path = os.path.join(subdir, filename)

                # 构造复制到 'all' 目录的路径
                destination = os.path.join(all_directory, filename)

                # 如果文件已存在，则跳过
                if not os.path.exists(destination):
                    shutil.copy(file_path, destination)
                    print(f'复制 {filename} 到 {all_directory}')
                else:
                    print(f'{filename} 已经存在，跳过复制')


def convert_path(original_path):
    # 使用正则表达式匹配并提取数字部分，然后构建新的路径
    new_path = re.sub(r'^\.\/data\/data_separation\\(\d+)\\', r'./result/picture/\1/', original_path)  # 替换数字部分并构建新路径
    new_path = re.sub(r'\.csv$', '.png', new_path)  # 替换.csv为.png

    # 创建目标路径中的目录（如果不存在的话）
    directory = os.path.dirname(new_path)
    if not os.path.exists(directory):
        os.makedirs(directory)  # 创建目录，包括任何必要的父目录

    return new_path


def get_files_with_extension_recursive(directory, extension):
    """
    获取指定目录及其子目录下所有具有特定后缀的文件名，返回一个列表。

    :param directory: 要搜索的文件夹路径
    :param extension: 文件后缀名，例如 '.txt' 或 '.csv'
    :return: 文件名列表
    """
    files = []
    for root, _, filenames in os.walk(directory):  # 遍历所有子目录和文件
        for filename in filenames:
            if filename.endswith(extension):
                # 获取文件的完整路径
                files.append(os.path.join(root, filename))
    return files


if __name__ == "__main__":
    # 1.数据构造部分
    reader_processor = data_reader.DataReaderProcessor(mode='bluetooth', save_dir="data/data_separation/")
    read_filenames = get_files_with_extension_recursive('./data/data_original', 'csv')
    for filename in read_filenames:
        reader_processor.process_file(filename)

    # 2.数据分析部分
    data_filenames = get_files_with_extension_recursive('./data/data_separation', '.csv')
    for filename in data_filenames:
        file_data = pd.read_csv(filename)
        signal_processor = data_filter.SignalProcessorFFT(file_data, defer_plotting=True)

        # 2.1滤波的应用
        signal_processor.band_pass_filter(5,40, plot=False)
        # signal_processor.notch_filter(notch_freq=50, Q=30)  # 假设需要去除 50Hz 噪声
        # signal_processor.notch_filter(notch_freq=100, Q=30)  # 可添加多个陷波滤波器

        # 2.2存储图像
        picture_name = convert_path(filename)
        signal_processor.save_comparison_plot(picture_name, custom_titles={
            'comparison_signal': "Original vs. Final Filtered Signal",
            'comparison_spectrum': "Original vs. Final Filtered Frequency Spectrum"
        })

    # 3.数据汇总
    directory = './result/picture/'  # 可以替换为需要遍历的目录路径
    copy_png_files_to_all_directory(directory)
