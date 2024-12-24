import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from pandas.plotting import parallel_coordinates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#加载数据的函数
def LoadData():
    #sources数据类型为（1084*6），每行数据代表一个帧的数据特征
    sources=np.array(pd.read_excel("../DataFile/result.xlsx",sheet_name="Sheet1"))
    standardScaler = StandardScaler()
    return standardScaler.fit_transform(sources)

#计算距离
def CalculateDistance(x, y):
    return np.linalg.norm(x - y)

#初始化中心点
def InitializeCeter(sources, K):
    indexs=np.random.choice(sources.shape[0], K, replace=False)
    return sources[indexs]

#更新样本标签
def UpdateLabel(dataset, center):
    newLabel = np.zeros(dataset.shape[0], dtype=np.int32)
    meanLoss = 0

    for i, data in enumerate(dataset):
        distance = np.array([CalculateDistance(data, c) for c in center])
        newLabel[i] = distance.argmin()
        meanLoss += distance[newLabel[i]]

    meanLoss /= dataset.shape[0]

    return newLabel, meanLoss

#计算簇
def CalculateClusters(dataset, label, K):
    clusters = [[] for i in range(K)]
    for i, x in enumerate(label):
        clusters[x].append(dataset[i])
    return [np.array(cluster) for cluster in clusters]

#更新中心点为每个簇内样本的均值
def CalculateCenter(clusters):
    #生成各个簇的中心点
    center = [[np.mean(cluster[:,i]) for i in range(cluster.shape[1])] for cluster in clusters if cluster.size > 0]
    return np.array(center)

#判断收敛
def ISEqual(label, newLabel):
    for i in range(label.shape[0]):
        if label[i] != newLabel[i]:
            return False
    return True

#Kmeans主算法，采用轮廓系数作为评估指标
def Kmeans(dataset, K):
    iteration = 0
    center = InitializeCeter(dataset, K)
    label, meanLoss = UpdateLabel(dataset, center)

    while True:
        iteration += 1
        clusters = CalculateClusters(dataset, label, K)
        center = CalculateCenter(clusters)
        newLabel, meanLoss = UpdateLabel(dataset, center)
        if not ISEqual(label, newLabel):
            label = newLabel
        else:
            break

    score = silhouette_score(dataset, newLabel)
    print('K={0}, 轮廓系数={1}'.format(K, score))
    return newLabel, score

#寻优算法，寻找最优的K值
def Search(dataset):
    bestK = None
    bestScore = 0
    bestLabel = None
    maxK=20
    for k in range(2, maxK):
        label, score = Kmeans(dataset, k)
        if score > bestScore:
            bestK = k
            bestScore = score
            bestLabel = label

    print('最优K值为：{0}, 轮廓系数为：{1}'.format(bestK, bestScore))

    return bestLabel,bestK

#加载数据
dataset=LoadData()
#插入数据的序号，便于后续处理
dataset=np.insert(dataset,0,[i for i in range(dataset.shape[0])],axis=1)

#划分训练集和测试集
x_train, x_test = train_test_split(dataset, test_size=0.2)
#寻找最优参数
labels,K=Search(x_train[:,1:])

#在测试数据集上测试模型
testLabels,scores=Kmeans(x_test[:,1:],K)
print(f"模型在测试数据集上的轮廓系数为{silhouette_score(x_test[:,1:],testLabels)}")

#输出测试数据点对应的标签，以[数据点序号  标签]的形式输出
print("测试数据每条对应的标签如下：")
testLabels=testLabels.reshape(testLabels.shape[0],1)
testLabels=np.insert(testLabels,0,x_test[:,0],axis=1)
print(testLabels)

#特征数据可视化
#删除序号列
x_test=np.delete(x_test,0,axis=1)
#添加标签列
x_test=np.insert(x_test,0,testLabels[:,1],axis=1)
x_test=pd.DataFrame(x_test)
plt.figure('Kmeans六维数据可视化')
plt.title('parallel_coordinates')
parallel_coordinates(x_test, 0)
plt.show()