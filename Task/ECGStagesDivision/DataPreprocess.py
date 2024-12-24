# 数据预处理，包括噪声和缺值处理
import numpy as np
from scipy.signal import butter, filtfilt


# 设计带通滤波器，数据的采样频率是200Hz
def bandpass_filter(signal, lowcut=0.5, highcut=50, fs=200, order=1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# 去除信号中的缺失值
def fill_missing_values(signal):
    # 检测缺失值（例如NaN）
    if np.any(np.isnan(signal)):
        # 使用线性插值填充缺失值
        nans = np.isnan(signal)
        x = np.arange(len(signal))
        signal[nans] = np.interp(x[nans], x[~nans], signal[~nans])

    return signal

#这是验证好坏
#这是测试集



#dahsadjkd
