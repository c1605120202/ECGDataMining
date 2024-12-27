import numpy as np
# import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, f1_score, \
    average_precision_score

from Task.ECGStagesDivision.Main import compute_features, LSTM_TIME_STEPS


# 评估函数
def evaluate(final_model, val_data, val_label):
    # 计算训练集上的特征
    X_val = compute_features(val_data)

    # 处理数据格式，使其符合lstm输入要求
    X_val = X_val.reshape(X_val.shape[0], LSTM_TIME_STEPS, int(X_val.shape[1] / LSTM_TIME_STEPS))  # 设置时间步长为3

    y_pred_prob = final_model.predict(X_val)  # 预测的概率矩阵
    y_pred = np.argmax(y_pred_prob, axis=1)

    print(f"模型在测试集上的准确率为：{accuracy_score(val_label, y_pred)}")

    # **2. 混淆矩阵**
    conf_matrix = confusion_matrix(val_label, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # **3. 绘制P-R曲线
    y_scores = y_pred_prob
    y_test = val_label

    # 计算每个类别的精确率和召回率
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(y_scores.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_test == i, y_scores[:, i])
        average_precision[i] = average_precision_score(y_test == i, y_scores[:, i], average='macro')

    plt.figure(figsize=(8, 6))
    for i in range(y_scores.shape[1]):
        plt.plot(recall[i], precision[i], label=f'Class {i} (AP = {average_precision[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve for multi-class')
    plt.legend()
    plt.show()

    # **4. F1 评价指标**
    f1 = f1_score(val_label, y_pred,average="macro")
    print(f"F1 Score: {f1:.4f}")

    # **5. Classification Report**
    print("Classification Report:")
    print(classification_report(val_label, y_pred))
