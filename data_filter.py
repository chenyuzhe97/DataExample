import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import os


class SignalProcessorFFT:
    def __init__(self, data, defer_plotting=False):
        """
        初始化 SignalProcessorFFT 对象。

        参数:
        data (pd.DataFrame): 包含时间和信号数据的 DataFrame
        defer_plotting (bool): 是否延迟绘图。若为 True，则不在每次滤波后绘图；否则，可根据滤波方法的 plot 参数控制绘图。
        """
        self.original_data = data.copy()
        self.data = data.copy()
        self.time = self.data['time'].values
        self.original_signal = self.data['value'].values.copy()
        self.fs = 1 / (self.time[1] - self.time[0])  # 采样频率
        self.defer_plotting = defer_plotting
        self.filter_history = []  # 记录所有应用的滤波器

        # 计算并存储原始信号的频域数据
        self.original_freq = fft(self.original_signal)
        self.current_frequencies = np.fft.fftfreq(len(self.original_signal), 1 / self.fs)

    def apply_filter(self, filter_type, cutoff_freqs):
        """
        应用指定类型的滤波器。

        参数:
        filter_type (str): 滤波器类型，支持 'high_pass', 'low_pass', 'band_pass', 'band_stop'
        cutoff_freqs (tuple or float): 滤波器的截止频率
                                       单个频率用于 'high_pass' 和 'low_pass',
                                       两个频率（下限和上限）用于 'band_pass' 和 'band_stop'

        返回:
        self
        """
        # 提取信号数据
        signal = self.data['value'].values

        # 执行 FFT 转换到频域
        N = len(signal)
        f_signal_freq = fft(signal)

        # 频率轴（包括正频率和负频率）
        frequencies = np.fft.fftfreq(N, 1 / self.fs)

        # 创建滤波器的掩码
        if filter_type == 'high_pass':
            cutoff = cutoff_freqs
            filter_mask = np.abs(frequencies) >= cutoff
            title_suffix = f"High-Pass (Cutoff = {cutoff} Hz)"
        elif filter_type == 'low_pass':
            cutoff = cutoff_freqs
            filter_mask = np.abs(frequencies) <= cutoff
            title_suffix = f"Low-Pass (Cutoff = {cutoff} Hz)"
        elif filter_type == 'band_pass':
            low, high = cutoff_freqs
            filter_mask = (np.abs(frequencies) >= low) & (np.abs(frequencies) <= high)
            title_suffix = f"Band-Pass (Cutoff = {low}-{high} Hz)"
        elif filter_type == 'band_stop':
            low, high = cutoff_freqs
            filter_mask = (np.abs(frequencies) < low) | (np.abs(frequencies) > high)
            title_suffix = f"Band-Stop (Cutoff = {low}-{high} Hz)"
        else:
            raise ValueError("Unsupported filter type. Choose from 'high_pass', 'low_pass', 'band_pass', 'band_stop'.")

        # 应用滤波器
        f_signal_freq_filtered = f_signal_freq * filter_mask

        # 反向 FFT 转换回时域
        filtered_signal = np.real(ifft(f_signal_freq_filtered))

        # 更新 DataFrame 中的信号
        self.data['value'] = filtered_signal

        # 存储当前滤波后的频域数据以供绘图使用
        self.filtered_freq = f_signal_freq_filtered
        self.title_suffix = title_suffix

        # 记录滤波器应用历史
        self.filter_history.append({
            'type': filter_type,
            'cutoff_freqs': cutoff_freqs,
            'title_suffix': title_suffix
        })

        return self

    def plot_signals(self, custom_titles=None):
        """
        绘制原始信号、当前滤波后信号及其频谱。

        参数:
        custom_titles (dict, optional): 包含自定义标题的字典。可包含以下键：
            - 'original_signal': 原始信号的标题
            - 'filtered_signal': 滤波后信号的标题
            - 'original_spectrum': 原始频谱的标题
            - 'filtered_spectrum': 滤波后频谱的标题
        """
        if custom_titles is None:
            custom_titles = {}

        original_signal = self.original_signal
        filtered_signal = self.data['value'].values
        frequencies = self.current_frequencies
        original_freq = self.original_freq
        filtered_freq = self.filtered_freq
        N = len(original_signal)

        plt.figure(figsize=(16, 12))

        # 原始信号
        plt.subplot(4, 1, 1)
        plt.plot(self.time, original_signal)
        title = custom_titles.get('original_signal', "Original Signal")
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        # 滤波后信号
        plt.subplot(4, 1, 2)
        plt.plot(self.time, filtered_signal)
        title = custom_titles.get('filtered_signal', f"Filtered Signal ({self.title_suffix})")
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        # 原始频谱
        plt.subplot(4, 1, 3)
        plt.plot(frequencies[:N // 2], np.abs(original_freq)[:N // 2])
        title = custom_titles.get('original_spectrum', "Original Frequency Spectrum")
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")

        # 滤波后频谱
        plt.subplot(4, 1, 4)
        plt.plot(frequencies[:N // 2], np.abs(filtered_freq)[:N // 2])
        title = custom_titles.get('filtered_spectrum', f"Filtered Frequency Spectrum ({self.title_suffix})")
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")

        # 调整布局
        plt.tight_layout()
        plt.show()

    def plot_comparison(self, custom_titles=None):
        """
        绘制原始信号与最终滤波信号的对比图，包括频谱对比。

        参数:
        custom_titles (dict, optional): 包含自定义标题的字典。可包含以下键：
            - 'comparison_signal': 比较信号的标题
            - 'comparison_spectrum': 比较频谱的标题
        """
        if custom_titles is None:
            custom_titles = {}

        original_signal = self.original_signal
        final_signal = self.data['value'].values
        N = len(original_signal)
        frequencies = self.current_frequencies

        # 原始频域信号
        original_freq = self.original_freq
        # 最终频域信号
        final_freq = self.filtered_freq

        plt.figure(figsize=(16, 12))

        # 原始信号 vs 最终信号
        plt.subplot(2, 1, 1)
        plt.plot(self.time, original_signal, label='Original Signal')
        plt.plot(self.time, final_signal, label='Final Filtered Signal', alpha=0.7)
        title = custom_titles.get('comparison_signal', "Original vs. Final Filtered Signal")
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend()

        # 原始频谱 vs 最终频谱
        plt.subplot(2, 1, 2)
        plt.plot(frequencies[:N // 2], np.abs(original_freq)[:N // 2], label='Original Spectrum')
        plt.plot(frequencies[:N // 2], np.abs(final_freq)[:N // 2], label='Final Filtered Spectrum', alpha=0.7)
        title = custom_titles.get('comparison_spectrum', "Original vs. Final Filtered Frequency Spectrum")
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.legend()

        # 调整布局
        plt.tight_layout()
        plt.show()

    def save_initial_plot(self, save_path, custom_titles=None):
        """
        保存原始信号及其频谱图到指定路径。

        参数:
        save_path (str): 保存图像的文件路径（包括文件名和扩展名，如 'path/to/initial_plot.png'）
        custom_titles (dict, optional): 包含自定义标题的字典。可包含以下键：
            - 'original_signal': 原始信号的标题
            - 'original_spectrum': 原始频谱的标题
        """
        if custom_titles is None:
            custom_titles = {}

        original_signal = self.original_signal
        original_freq = self.original_freq
        frequencies = self.current_frequencies
        N = len(original_signal)

        plt.figure(figsize=(16, 8))

        # 原始信号
        plt.subplot(2, 1, 1)
        plt.plot(self.time, original_signal)
        title = custom_titles.get('original_signal', "Original Signal")
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        # 原始频谱
        plt.subplot(2, 1, 2)
        plt.plot(frequencies[:N // 2], np.abs(original_freq)[:N // 2])
        title = custom_titles.get('original_spectrum', "Original Frequency Spectrum")
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")

        # 调整布局
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Initial plot saved to {save_path}")

    def save_final_plot(self, save_path, custom_titles=None):
        """
        保存当前滤波后的信号及其频谱图到指定路径。

        参数:
        save_path (str): 保存图像的文件路径（包括文件名和扩展名，如 'path/to/final_plot.png'）
        custom_titles (dict, optional): 包含自定义标题的字典。可包含以下键：
            - 'filtered_signal': 滤波后信号的标题
            - 'filtered_spectrum': 滤波后频谱的标题
        """
        if custom_titles is None:
            custom_titles = {}

        filtered_signal = self.data['value'].values
        filtered_freq = self.filtered_freq
        frequencies = self.current_frequencies
        N = len(filtered_signal)

        plt.figure(figsize=(16, 8))

        # 滤波后信号
        plt.subplot(2, 1, 1)
        plt.plot(self.time, filtered_signal)
        title = custom_titles.get('filtered_signal', f"Filtered Signal ({self.title_suffix})")
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        # 滤波后频谱
        plt.subplot(2, 1, 2)
        plt.plot(frequencies[:N // 2], np.abs(filtered_freq)[:N // 2])
        title = custom_titles.get('filtered_spectrum', f"Filtered Frequency Spectrum ({self.title_suffix})")
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")

        # 调整布局
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Final plot saved to {save_path}")

    def save_comparison_plot(self, save_path, custom_titles=None):
        """
        保存原始信号与最终滤波信号的对比图，包括频谱对比到指定路径。

        参数:
        save_path (str): 保存图像的文件路径（包括文件名和扩展名，如 'path/to/comparison_plot.png'）
        custom_titles (dict, optional): 包含自定义标题的字典。可包含以下键：
            - 'comparison_signal': 比较信号的标题
            - 'comparison_spectrum': 比较频谱的标题
        """
        if custom_titles is None:
            custom_titles = {}

        original_signal = self.original_signal
        final_signal = self.data['value'].values
        N = len(original_signal)
        frequencies = self.current_frequencies

        # 原始频域信号
        original_freq = self.original_freq
        # 最终频域信号
        final_freq = self.filtered_freq

        plt.figure(figsize=(16, 12))

        # 原始信号 vs 最终信号
        plt.subplot(2, 1, 1)
        plt.plot(self.time, original_signal, label='Original Signal')
        plt.plot(self.time, final_signal, label='Final Filtered Signal', alpha=0.7)
        title = custom_titles.get('comparison_signal', "Original vs. Final Filtered Signal")
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend()

        # 原始频谱 vs 最终频谱
        plt.subplot(2, 1, 2)
        plt.plot(frequencies[:N // 2], np.abs(original_freq)[:N // 2], label='Original Spectrum')
        plt.plot(frequencies[:N // 2], np.abs(final_freq)[:N // 2], label='Final Filtered Spectrum', alpha=0.7)
        title = custom_titles.get('comparison_spectrum', "Original vs. Final Filtered Frequency Spectrum")
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.legend()

        # 调整布局
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Comparison plot saved to {save_path}")

    def high_pass_filter(self, cutoff_freq, plot=True, custom_titles=None):
        """
        应用高通滤波器。

        参数:
        cutoff_freq (float): 高通滤波器的截止频率
        plot (bool): 是否绘制滤波结果（仅当 defer_plotting=False 时有效）
        custom_titles (dict, optional): 自定义标题的字典

        返回:
        self
        """
        self.apply_filter('high_pass', cutoff_freq)
        if not self.defer_plotting and plot:
            self.plot_signals(custom_titles)
        return self

    def low_pass_filter(self, cutoff_freq, plot=True, custom_titles=None):
        """
        应用低通滤波器。

        参数:
        cutoff_freq (float): 低通滤波器的截止频率
        plot (bool): 是否绘制滤波结果（仅当 defer_plotting=False 时有效）
        custom_titles (dict, optional): 自定义标题的字典

        返回:
        self
        """
        self.apply_filter('low_pass', cutoff_freq)
        if not self.defer_plotting and plot:
            self.plot_signals(custom_titles)
        return self

    def band_pass_filter(self, low_cutoff, high_cutoff, plot=True, custom_titles=None):
        """
        应用带通滤波器。

        参数:
        low_cutoff (float): 带通滤波器的低截止频率
        high_cutoff (float): 带通滤波器的高截止频率
        plot (bool): 是否绘制滤波结果（仅当 defer_plotting=False 时有效）
        custom_titles (dict, optional): 自定义标题的字典

        返回:
        self
        """
        self.apply_filter('band_pass', (low_cutoff, high_cutoff))
        if not self.defer_plotting and plot:
            self.plot_signals(custom_titles)
        return self

    def band_stop_filter(self, lower_cutoff, upper_cutoff, plot=True, custom_titles=None):
        """
        应用带阻滤波器。

        参数:
        lower_cutoff (float): 带阻滤波器的下截止频率
        upper_cutoff (float): 带阻滤波器的上截止频率
        plot (bool): 是否绘制滤波结果（仅当 defer_plotting=False 时有效）
        custom_titles (dict, optional): 自定义标题的字典

        返回:
        self
        """
        self.apply_filter('band_stop', (lower_cutoff, upper_cutoff))
        if not self.defer_plotting and plot:
            self.plot_signals(custom_titles)
        return self

    def notch_filter(self, notch_freq, Q=30, plot=True, custom_titles=None):
        """
        应用陷波滤波器（Notch Filter）。

        参数:
        notch_freq (float): 陷波滤波器的目标频率
        Q (float): 品质因数，定义带宽。Q 值越高，带宽越窄
        plot (bool): 是否绘制滤波结果（仅当 defer_plotting=False 时有效）
        custom_titles (dict, optional): 自定义标题的字典

        返回:
        self
        """
        # 计算带阻滤波器的带宽
        bandwidth = notch_freq / Q
        low_cutoff = notch_freq - bandwidth / 2
        high_cutoff = notch_freq + bandwidth / 2

        # 确保截止频率在合理范围内
        if low_cutoff < 0:
            low_cutoff = 0
        if high_cutoff > self.fs / 2:
            high_cutoff = self.fs / 2

        # 应用带阻滤波器
        self.apply_filter('band_stop', (low_cutoff, high_cutoff))

        # 如果需要自定义标题
        if custom_titles is None:
            custom_titles = {}
        title_suffix = f"Notch Filter (Freq = {notch_freq} Hz, Q = {Q})"
        self.title_suffix = title_suffix

        if not self.defer_plotting and plot:
            # 更新自定义标题
            if 'filtered_signal' not in custom_titles:
                custom_titles['filtered_signal'] = f"Filtered Signal ({title_suffix})"
            if 'filtered_spectrum' not in custom_titles:
                custom_titles['filtered_spectrum'] = f"Filtered Frequency Spectrum ({title_suffix})"
            self.plot_signals(custom_titles)

        # 记录滤波器应用历史
        self.filter_history.append({
            'type': 'notch_filter',
            'notch_freq': notch_freq,
            'Q': Q,
            'bandwidth': bandwidth,
            'title_suffix': title_suffix
        })

        return self

    def reset_signal(self):
        """
        重置信号为原始信号。

        返回:
        self
        """
        self.data = self.original_data.copy()
        # 重新计算频域数据
        self.filtered_freq = self.original_freq.copy()
        return self

    def get_data(self):
        """
        获取当前的 DataFrame 数据。

        返回:
        pd.DataFrame: 当前的 DataFrame
        """
        return self.data

    def save_initial_plot(self, save_path, custom_titles=None):
        """
        保存原始信号及其频谱图到指定路径。

        参数:
        save_path (str): 保存图像的文件路径（包括文件名和扩展名，如 'path/to/initial_plot.png'）
        custom_titles (dict, optional): 包含自定义标题的字典。可包含以下键：
            - 'original_signal': 原始信号的标题
            - 'original_spectrum': 原始频谱的标题
        """
        if custom_titles is None:
            custom_titles = {}

        original_signal = self.original_signal
        original_freq = self.original_freq
        frequencies = self.current_frequencies
        N = len(original_signal)

        plt.figure(figsize=(16, 8))

        # 原始信号
        plt.subplot(2, 1, 1)
        plt.plot(self.time, original_signal)
        title = custom_titles.get('original_signal', "Original Signal")
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        # 原始频谱
        plt.subplot(2, 1, 2)
        plt.plot(frequencies[:N // 2], np.abs(original_freq)[:N // 2])
        title = custom_titles.get('original_spectrum', "Original Frequency Spectrum")
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")

        # 调整布局
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Initial plot saved to {save_path}")

    def save_final_plot(self, save_path, custom_titles=None):
        """
        保存当前滤波后的信号及其频谱图到指定路径。

        参数:
        save_path (str): 保存图像的文件路径（包括文件名和扩展名，如 'path/to/final_plot.png'）
        custom_titles (dict, optional): 包含自定义标题的字典。可包含以下键：
            - 'filtered_signal': 滤波后信号的标题
            - 'filtered_spectrum': 滤波后频谱的标题
        """
        if custom_titles is None:
            custom_titles = {}

        filtered_signal = self.data['value'].values
        filtered_freq = self.filtered_freq
        frequencies = self.current_frequencies
        N = len(filtered_signal)

        plt.figure(figsize=(16, 8))

        # 滤波后信号
        plt.subplot(2, 1, 1)
        plt.plot(self.time, filtered_signal)
        title = custom_titles.get('filtered_signal', f"Filtered Signal ({self.title_suffix})")
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        # 滤波后频谱
        plt.subplot(2, 1, 2)
        plt.plot(frequencies[:N // 2], np.abs(filtered_freq)[:N // 2])
        title = custom_titles.get('filtered_spectrum', f"Filtered Frequency Spectrum ({self.title_suffix})")
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")

        # 调整布局
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Final plot saved to {save_path}")

    def save_comparison_plot(self, save_path, custom_titles=None):
        """
        保存原始信号与最终滤波信号的对比图，包括频谱对比到指定路径。

        参数:
        save_path (str): 保存图像的文件路径（包括文件名和扩展名，如 'path/to/comparison_plot.png'）
        custom_titles (dict, optional): 包含自定义标题的字典。可包含以下键：
            - 'comparison_signal': 比较信号的标题
            - 'comparison_spectrum': 比较频谱的标题
        """
        if custom_titles is None:
            custom_titles = {}

        original_signal = self.original_signal
        final_signal = self.data['value'].values
        N = len(original_signal)
        frequencies = self.current_frequencies

        # 原始频域信号
        original_freq = self.original_freq
        # 最终频域信号
        final_freq = self.filtered_freq

        plt.figure(figsize=(16, 12))

        # 原始信号 vs 最终信号
        plt.subplot(2, 1, 1)
        plt.plot(self.time, original_signal, label='Original Signal')
        plt.plot(self.time, final_signal, label='Final Filtered Signal', alpha=0.7)
        title = custom_titles.get('comparison_signal', "Original vs. Final Filtered Signal")
        plt.title(title)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend()

        # 原始频谱 vs 最终频谱
        plt.subplot(2, 1, 2)
        plt.plot(frequencies[:N // 2], np.abs(original_freq)[:N // 2], label='Original Spectrum')
        plt.plot(frequencies[:N // 2], np.abs(final_freq)[:N // 2], label='Final Filtered Spectrum', alpha=0.7)
        title = custom_titles.get('comparison_spectrum', "Original vs. Final Filtered Frequency Spectrum")
        plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.legend()

        # 调整布局
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Comparison plot saved to {save_path}")

    def reset_signal(self):
        """
        重置信号为原始信号。

        返回:
        self
        """
        self.data = self.original_data.copy()
        # 重新计算频域数据
        self.filtered_freq = self.original_freq.copy()
        return self

    def get_data(self):
        """
        获取当前的 DataFrame 数据。

        返回:
        pd.DataFrame: 当前的 DataFrame
        """
        return self.data


# 示例使用
if __name__ == "__main__":
    # 读取数据
    # 请确保 CSV 文件包含 'time' 和 'value' 两列
    file_data = pd.read_csv('./data/data_origin/example1.csv')

    # 创建 SignalProcessorFFT 对象
    # 若希望延迟绘图，设置 defer_plotting=True
    processor = SignalProcessorFFT(file_data, defer_plotting=True)

    # 定义自定义标题
    custom_titles1 = {
        'filtered_signal': "Custom High-Pass Filtered Signal",
        'filtered_spectrum': "Custom High-Pass Filtered Frequency Spectrum"
    }

    custom_titles2 = {
        'filtered_signal': "Custom Band-Stop Filtered Signal",
        'filtered_spectrum': "Custom Band-Stop Filtered Frequency Spectrum"
    }

    # 定义保存路径
    initial_plot_path = './plots/initial_plot.png'  # 请根据需要修改路径
    final_plot_path = './plots/final_plot.png'  # 请根据需要修改路径
    comparison_plot_path = './plots/comparison_plot.png'  # 请根据需要修改路径

    # 确保保存目录存在
    os.makedirs(os.path.dirname(initial_plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(final_plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(comparison_plot_path), exist_ok=True)

    # 保存初始信号及其频谱图
    processor.save_initial_plot(initial_plot_path, custom_titles={
        'original_signal': "Original Signal",
        'original_spectrum': "Original Frequency Spectrum"
    })

    # 示例1：延迟绘图，最终绘制对比图
    (processor.high_pass_filter(20, plot=False)
    .band_stop_filter(45, 55, plot=False)
    .band_stop_filter(95, 105, plot=False)
    .plot_comparison(custom_titles={
        'comparison_signal': "Original vs. Final Filtered Signal",
        'comparison_spectrum': "Original vs. Final Filtered Frequency Spectrum"
    }))

    # 保存最终滤波后的信号及其频谱图
    processor.save_final_plot(final_plot_path, custom_titles={
        'filtered_signal': "Final Filtered Signal",
        'filtered_spectrum': "Final Filtered Frequency Spectrum"
    })

    # 保存对比图
    processor.save_comparison_plot(comparison_plot_path, custom_titles={
        'comparison_signal': "Original vs. Final Filtered Signal",
        'comparison_spectrum': "Original vs. Final Filtered Frequency Spectrum"
    })

    # 示例2：即时绘图（不延迟）
    processor.reset_signal()
    processor.defer_plotting = False  # 改变绘图模式
    processor.band_stop_filter(30, 40, plot=True, custom_titles=custom_titles2) \
        .low_pass_filter(20, plot=True)

    # 示例3：使用陷波滤波器
    processor.reset_signal()
    processor.defer_plotting = True  # 延迟绘图
    processor.high_pass_filter(0.3, plot=False)
    processor.band_pass_filter(0, 90, plot=False)
    processor.notch_filter(notch_freq=50, Q=30)  # 假设需要去除 50Hz 噪声
    processor.notch_filter(notch_freq=100, Q=30)  # 可添加多个陷波滤波器
    processor.plot_comparison(custom_titles={
        'comparison_signal': "Original vs. Final Filtered Signal with Notch Filters",
        'comparison_spectrum': "Original vs. Final Filtered Frequency Spectrum with Notch Filters"
    })

    # 保存对比图（带陷波滤波器）
    comparison_plot_notch_path = './plots/comparison_plot_notch.png'  # 请根据需要修改路径
    os.makedirs(os.path.dirname(comparison_plot_notch_path), exist_ok=True)
    processor.save_comparison_plot(comparison_plot_notch_path, custom_titles={
        'comparison_signal': "Original vs. Final Filtered Signal with Notch Filters",
        'comparison_spectrum': "Original vs. Final Filtered Frequency Spectrum with Notch Filters"
    })

    # 获取最终的过滤数据
    filtered_data = processor.get_data()

    # 可选：保存过滤后的数据
    # filtered_data.to_csv('filtered_example1.csv', index=False)
