import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#数据标签
labels=np.loadtxt("../DataFile/label_profusion.txt")
#sources数据类型为（1084*6），每行数据代表一个帧的数据特征
sources=np.array(pd.read_excel("../DataFile/result.xlsx",sheet_name="Sheet1"))

#划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(sources, labels, test_size=0.2)

# 4.机器学习--knn+cv
# 4.1 实例化一个估计器
estimator = KNeighborsClassifier()
# 4.2 调用gridsearchCV
param_grid = {"n_neighbors": [1, 3, 5]}
estimator = GridSearchCV(estimator, param_grid=param_grid, cv=3)
# print(estimator.best_params_)
# 4.3 模型训练
estimator.fit(x_train, y_train)

# 5.模型评估
# 5.1 基本评估方式
score = estimator.score(x_test, y_test)
print("最后预测的准确率为:\n", score)

y_predict = estimator.predict(x_test)
print("最后的预测值为:\n", y_predict)
print("预测值和真实值的对比情况:\n", y_predict == y_test)

# 5.2 使用交叉验证后的评估方式
print("在交叉验证中验证的最好结果:\n", estimator.best_score_)

#输出混淆矩阵
knn_matrix=confusion_matrix(y_test, y_predict)
print(f"KNN的混淆矩阵为：\n{knn_matrix}")