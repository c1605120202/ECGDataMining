import numpy as np
import scipy.stats as st
import pywt
import antropy as ant

from scipy.spatial.distance import cdist


# 计算Lyapunov指数
def calculate_lyapunov_exponent(ecg_data, delay=20, embedding_dim=5, max_iter=500, epsilon=1e-4):
    """
    计算 ECG 数据的最大 Lyapunov 指数。

    参数:
        ecg_data (array-like): 输入的 ECG 时间序列（必须预处理）。
        delay (int): 时间延迟，用于相空间重构。
        embedding_dim (int): 嵌入维度。
        max_iter (int): 最大迭代步数，用于跟踪扰动演化。
        epsilon (float): 初始扰动的阈值。

    返回:
        float: 最大 Lyapunov 指数。
    """

    # --- 1. 相空间重构 ---
    def reconstruct_phase_space(data, delay, dim):
        n_vectors = len(data) - (dim - 1) * delay
        phase_space = np.empty((n_vectors, dim))
        for i in range(dim):
            phase_space[:, i] = data[i * delay: i * delay + n_vectors]
        return phase_space

    phase_space = reconstruct_phase_space(ecg_data, delay, embedding_dim)
    n_points = len(phase_space)

    # --- 2. 初始化最近邻点 ---
    distances = cdist(phase_space, phase_space)  # 所有点之间的距离矩阵
    np.fill_diagonal(distances, np.inf)  # 自身距离设为无穷大
    nearest_neighbors = np.argmin(distances, axis=1)  # 最近邻点索引

    # --- 3. Lyapunov 指数估算 ---
    lyapunov_exponents = []
    for i in range(n_points - max_iter):
        ref_idx = i
        neighbor_idx = nearest_neighbors[ref_idx]

        # 初始距离
        initial_dist = np.linalg.norm(phase_space[ref_idx] - phase_space[neighbor_idx])
        if initial_dist < epsilon:  # 排除距离过小的点对
            continue

        # 跟踪扰动演化
        distances = []
        for k in range(max_iter):
            ref_idx_k = ref_idx + k
            neighbor_idx_k = neighbor_idx + k
            if neighbor_idx_k >= n_points or ref_idx_k >= n_points:
                break
            current_dist = np.linalg.norm(
                phase_space[ref_idx_k] - phase_space[neighbor_idx_k]
            )
            distances.append(current_dist / initial_dist)  # 归一化距离增长

        # 对数增长率
        log_growth_rates = np.log(distances)
        avg_growth_rate = np.mean(log_growth_rates)
        lyapunov_exponents.append(avg_growth_rate)

    # 返回平均 Lyapunov 指数，无则返回0
    return np.mean(lyapunov_exponents) if lyapunov_exponents else 0


# 计算吸引子维数（关联维数）
def correlation_dimension(time_series, m=2, tau=1, max_dist=None):
    """
    计算ECG信号的吸引子维数（使用Grassberger-Procaccia算法）
    :param time_series: 输入的ECG时间序列
    :param m: 嵌入维数
    :param tau: 延迟时间
    :param max_dist: 最大距离，如果为None则计算信号中最大点对距离
    :return: 吸引子维数
    """
    N = len(time_series)

    # 构建嵌入空间
    vectors = np.array([time_series[i:i + m * tau:tau] for i in range(N - m * tau)])

    # 计算所有点对之间的欧几里得距离
    dists = np.linalg.norm(vectors[:, None] - vectors, axis=2)

    if max_dist is None:
        max_dist = np.max(dists)  # 默认最大距离

    # 计算关联函数C(r)
    r = np.logspace(-2, np.log10(max_dist), 100)  # 设置距离尺度r
    correlation = np.array([np.sum(dists < ri) / (N * (N - 1)) for ri in r])

    # 拟合并计算吸引子维数：D2 = lim r->0 [log(C(r)) / log(r)]
    D2 = np.polyfit(np.log(r), np.log(correlation), 1)[0]
    return D2


# 计算Petrosian分形维数
def petrosian_fd(time_series):
    N = len(time_series)

    # 计算符号变化的次数（零交叉点）
    diff = np.diff(np.sign(time_series))  # 计算符号变化
    N_v = np.sum(diff != 0)  # 符号变化的次数

    # 计算 Petrosian 分形维数
    if N_v == 0:  # 处理特殊情况，避免除以 0
        return 0
    return np.log10(N) / (np.log10(N) + np.log10(N / N_v))


# 计算小波系数
# coeffs 是一个 列表，包含了多层次的小波系数。具体来说，这个列表的第一个元素是低频部分
# （近似系数），后续的元素是不同分解层次的高频部分（细节系数）。
def compute_wavelet_coefficients(signal, wavelet='db4', level=3):
    """
    计算ECG信号的小波系数
    :param signal: 输入ECG信号（1D数组）
    :param wavelet: 选择的小波函数（如'db4', 'sym4'等）
    :param level: 小波分解的层数
    :return: 小波分解的系数列表
    """
    # 使用小波变换进行分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs


# 拿到数据特征（每次输入一个二维数组signal_data）
# 返回：[小波系数（取近似系数部分），偏度，峰度，Lyapunov指数，吸引子维数（Fractal Dimension）
# Petrosian分形维数,均值,标准差,中值] 组成的特征集合
# 返回：numpy数组类型
# def getFeature(signal_datas):
#     # 分别表示数据的行和列数
#     lines = signal_datas.shape[0]
#     rows = signal_datas.shape[1]
#     results = []
#     for j in range(lines):
#         signal_data = signal_datas[j]
#         # 存放单条数据的特征
#         result = []
#         # result=list(compute_wavelet_coefficients(signal_data)[0])     #小波系数(取近似系数部分）
#         result.append(st.skew(signal_data))  # 偏度
#         result.append(st.kurtosis(signal_data))  # 峰度
#         # result.append(lyapunov_exponent(signal_data))  # Lyapunov指数
#         # result.append(correlation_dimension(signal_data))  # 吸引子维数
#         result.append(petrosian_fd(signal_data))  # Petrosian分形维数
#         result.append(np.average(signal_data))  # 均值
#         result.append(np.std(signal_data))  # 标准差
#         result.append(np.median(signal_data))  # 中值
#         results.append(result)
#         print(f"第{j}个：  {result}")
#         # if j == 99:
#         #     break
#
#     return np.array(results)

def getFeature(signal_data):
    result = []
    result.append(np.average(signal_data))  # 均值
    result.append(np.std(signal_data))  # 标准差
    result.append(petrosian_fd(signal_data))  # Petrosian分形维数
    result.append(np.median(signal_data))  # 中值
    result.append(st.skew(signal_data))  # 偏度
    result.append(st.kurtosis(signal_data))  # 峰度
    # result.append(calculate_lyapunov_exponent(signal_data))  # Lyapunov指数
    # result.append(correlation_dimension(signal_data))  # 吸引子维数
    result.append(ant.svd_entropy(signal_data,order=3,delay=1,normalize=True))       #计算svd分解熵
    result.append(ant.sample_entropy(signal_data,order=3,delay=1,normalize=True))      #计算样本熵
    return result
