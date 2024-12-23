import numpy as np
import pandas as pd
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree

#数据标签
labels=np.loadtxt("../DataFile/label_profusion.txt")
#sources数据类型为（1084*6），每行数据代表一个帧的数据特征
sources=np.array(pd.read_excel("../DataFile/result.xlsx",sheet_name="Sheet1"))

#划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(sources, labels, test_size=0.3)

#最好的最大深度，最高的准确性
best_max_depth,best_acu=0,0
#遍历不同深度时，判定决策树的预测性能。取准确率最高时的深度为最优的深度
for i in range(1,22):
    clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0,max_depth=i)
    clf = clf.fit(x_train,y_train)
    acu_score=clf.score(x_test,y_test)
    print(f"深度 {i}     准确率 {acu_score}")
    if acu_score>best_acu:
        best_acu=acu_score
        best_max_depth=i
print(f"当最大深度为{best_max_depth}时，模型预测准确率最高，为{best_acu}")

#代入最优深度训练决策树
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=best_max_depth)
clf = clf.fit(sources, labels)

#特征名和标签名
feature_names=["Mean","StdDeviation","Median","Petrosian","Skewness","Kurtosis"]
target_names=["0","1","2","3","4"]
#绘制决策树，并生成pdf文档
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=feature_names,class_names=target_names)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("../DataFile/EEG_clf1.pdf")

#输出混淆矩阵
cf_matrix=confusion_matrix(labels, clf.predict(sources) )
print(f"决策树的混淆矩阵为：\n{cf_matrix}")

