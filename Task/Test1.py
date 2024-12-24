import numpy as np

# 假设你有一个一维的时间序列数据
data = np.array([[1,2],[3,4],[5,6]])  # 这里your_2d_data_here是你的二维数据
print(data)
# 确定时间步长
look_back = 3  # 例如，使用过去3个时间点的数据来预测下一个时间点

# 创建数据集
dataX, dataY = [], []
for i in range(len(data) - look_back - 1):
    a = data[i:(i + look_back)]  # 选择时间步长内的数据
    dataX.append(a)
    dataY.append(data[i + look_back])  # 预测的目标值

# 将数据X转换为三维数组
dataX = np.array(dataX)
dataX = dataX.reshape(dataX.shape[0], look_back, 1)  # 重塑为三维数组
print(dataX)
# 现在dataX就是LSTM的输入数据，其形状为（样本数，时间步长，特征数）