import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from itertools import combinations

#读入数据源
data=np.array(np.loadtxt("../DataFile/EEG.txt"))
leap=30*125     #表示每帧中含有的数据量

#将全部的数据进行分组，分成1084*（30*125）
i,EEG_data=1,[]
while i*leap<=data.__len__():
    temp=data[(i-1)*leap:i*leap]
    EEG_data.append(temp)
    i+=1
#当数据不是整步长时，对最后一节数据进行单独分组
if i*leap-data.__len__()<leap:
    EEG_data.append(data[(i-1)*leap:data.__len__()])
EEG_data=np.array(EEG_data)

#使用皮尔逊相关系数和欧式距离来衡量信号之间的相似性和相异性
#用于计算各个信号对之间的相关性信息，返回一个包含相似性和相异性信息的字典列表
def calculate_similarity(data):
    num_signals = data.shape[0]
    signal_pairs = list(combinations(range(num_signals), 2))  # 生成信号对的组合
    similarity_results = []

    for pair in signal_pairs:
        signal1 = data[pair[0], :]
        signal2 = data[pair[1], :]

        # 计算皮尔逊相关系数
        corr, _ = pearsonr(signal1, signal2)

        # 计算欧氏距离
        dist = euclidean(signal1, signal2)

        similarity_results.append({
            'pair': pair,
            'correlation': corr,
            'distance': dist
        })
    return similarity_results

#计算数据帧的相似性和相异性参数
similarity_results = calculate_similarity(EEG_data)

#按照相关系数排序(绝对值），找到最相似的信号对
most_similar = sorted(similarity_results, key=lambda x: abs(x['correlation']), reverse=True)[:3]

# 按照欧氏距离排序，找到最相异的信号对
most_different = sorted(similarity_results, key=lambda x: x['distance'], reverse=True)[:3]

#输出相似相异性的列表
print("最相似的3对信号：")
for item in most_similar:
    print(f"信号对: {item['pair']}, 相关系数: {item['correlation']:.4f}, 欧氏距离: {item['distance']:.4f}")
print("\n最相异的3对信号：")
for item in most_different:
    print(f"信号对: {item['pair']}, 相关系数: {item['correlation']:.4f}, 欧氏距离: {item['distance']:.4f}")

#绘图函数
def plot_signals(data, pair, title):
    plt.figure(figsize=(10, 5))
    #画出子图1
    plt.subplot(2,1,1)
    #画出前两百个点
    plt.plot(data[pair[0], :][0:200])
    plt.ylim((-0.00015,0.00015))   #设置坐标轴的上下限
    plt.title(pair[0])
    #画出子图2
    plt.subplot(2,1,2)
    plt.plot(data[pair[1], :][0:200])
    plt.ylim((-0.00015,0.00015))
    plt.title(pair[1])
    plt.suptitle(title)
    plt.show()

# 可视化最相异的信号对
item = most_similar[0]
plot_signals(EEG_data, item['pair'], f"The most similar signal pair: {item['pair']} - Correlation Coefficient: {item['correlation']:.4f}")

# 可视化最相异的信号对
item = most_different[0]
plot_signals(EEG_data, item['pair'], f"The most different signal pair {item['pair']} - Euclidean Distance: {item['distance']:.4f}")
