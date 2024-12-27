import numpy as np
import scipy.stats as st

from scipy.spatial.distance import cdist
from collections import Counter

from sklearn.preprocessing import StandardScaler


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


# 计算svd熵

def svd_entropy(time_series):
    # 将时间序列转换为一个矩阵
    # 这里我们通过嵌入一个时间延迟来创建一个二维矩阵
    # 假设我们选择嵌入维度 m=3，时间延迟 d=1
    m = 3
    d = 1
    N = len(time_series)

    # 生成嵌入矩阵
    X = np.array([time_series[i:i + m] for i in range(N - m)])

    # 对嵌入矩阵进行SVD分解
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # 归一化奇异值
    S_norm = S / np.sum(S)

    # 计算熵
    entropy = -np.sum(S_norm * np.log(S_norm + np.finfo(float).eps))  # 加上eps防止log(0)

    return entropy


"""
   计算排列熵（Permutation Entropy）值

   time_series: 输入的时间序列数据（1D数组）
   m: 嵌入维度（通常为2或3）
   delay: 时间延迟（通常为1）
   """
def permutation_entropy(time_series, m=3, delay=1):
    N = len(time_series)

    if N <= m:
        raise ValueError("时间序列长度应大于嵌入维度m")

    # 生成嵌入子序列
    embedded_series = [time_series[i:i + m] for i in range(0, N - m + 1, delay)]

    # 计算每个子序列的排列模式
    permutations = []
    for subseq in embedded_series:
        # 获取排列顺序
        order = np.argsort(subseq)
        permutation = tuple(order)  # 将顺序转换为元组
        permutations.append(permutation)

    # 统计每个排列模式的出现频率
    permutation_counts = Counter(permutations)
    total_permutations = len(permutations)

    # 计算排列模式的概率分布
    probabilities = [count / total_permutations for count in permutation_counts.values()]

    # 计算排列熵
    nums = [p * np.log(p) for p in probabilities if p > 0]
    entropy = -np.sum(nums)

    return entropy


# 拿到数据特征（每次输入一个二维数组signal_data）
# 返回：[偏度，峰度， Petrosian分形维数,均值,标准差,中值] 组成的特征集合
# 返回：numpy数组类型
def getFeature(signal_datas):
    # 分别表示数据的行和列数
    lines = signal_datas.shape[0]
    rows = signal_datas.shape[1]
    results = []
    for j in range(lines):
        # 存放单条数据的特征
        result = []
        #200hz对应30s
        for i in range(30):
            signal_data = signal_datas[j,i * 200:(i + 1) * 200]
            result.append(np.average(signal_data))  # 均值
            result.append(np.std(signal_data))  # 标准差
            result.append(petrosian_fd(signal_data))  # Petrosian分形维数
            result.append(np.median(signal_data))  # 中值
            result.append(st.skew(signal_data))  # 偏度
            result.append(st.kurtosis(signal_data))  # 峰度
            result.append(permutation_entropy(signal_data))   #排列熵
            result.append(svd_entropy(signal_data))    #svd熵
            result.append(correlation_dimension(signal_data))    #吸引子维数
        results.append(result)
        print(f"{j}/{lines}    {result}")
    return np.array(results)


# 使用多进程计算原始数据的特征,并进行标准化处理
def compute_features(data):
    features = getFeature(data)
    features = np.array(features)

    print(features.shape)

    # 对特征进行标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    return X
