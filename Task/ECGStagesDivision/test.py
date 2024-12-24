from Task.ECGStagesDivision.DataPreprocess import *
from Task.ECGStagesDivision.FeatureExtract import *
import numpy as np
import random

from collections import Counter


def permutation_entropy(time_series, m=3, delay=1):
    """
    计算排列熵（Permutation Entropy）值

    time_series: 输入的时间序列数据（1D数组）
    m: 嵌入维度（通常为2或3）
    delay: 时间延迟（通常为1）
    """
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
    nums=[p * np.log(p) for p in probabilities if p > 0]
    entropy = -np.sum(nums)

    return entropy


a=[random.randint(0,20000) for i in range(9000)]
a=np.array(a)
print(permutation_entropy(a))
print(svd_entropy(a))