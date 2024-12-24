import multiprocessing

import numpy as np
import scipy.io
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential, load_model

from Task.ECGStagesDivision.DataPreprocess import *
from Task.ECGStagesDivision.FeatureExtract import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 文件根目录
DATA_PATH = "D:/Dataset/data"
LABEL_PATH = "D:/Dataset/label"
SUBJECT_NUM=100        #要读取的subject数量
CROSS_VALIDATE_NUM=5     #表示几则交叉验证
LSTM_TIME_STEPS=4     #算法时间步

# 定义LSTM模型函数
def create_lstm_model(input_shape, units=50, learning_rate=0.05):
    model = Sequential([
        LSTM(units, activation='tanh', input_shape=input_shape),
        Dense(5, activation='softmax')  # 多分类问题
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# 读取数据并进行预处理，返回纯净的数据
def load_data(index_list):
    print(f"参数序列： {index_list}")

    data, label = np.empty((0, 6000)), np.empty(0)
    for i in index_list:  # 读取按人划分的训练集
        print(i)
        # 读取X_1.txt作为标签文件
        with open(LabelPath(i), 'r') as file:
            data_list = file.readlines()
        # 去掉每行末尾的换行符
        data_list = [line.strip() for line in data_list]

        tmp = []  # 标签数据集合
        for k in range(0, len(data_list)):
            if data_list[k] == '':
                continue
            tmp.append(int(data_list[k]))
        tmp = tmp[0:-30]  # 舍弃掉后30个epoch的标签
        # 标签中跳过了4直接到5，由于需要生成one-hot编码所以需要进行转换
        label = np.concatenate((label, np.array([4 if num == 5 else num for num in tmp])))

        # 加载 .mat 文件作为数据文件
        mat_file = scipy.io.loadmat(DataPath(i))
        # 访问某个特定的变量（例如：变量名是 'data'）
        dt = mat_file['X2']
        data = np.concatenate((data, dt), axis=0)

        print(len(dt))
        print(len(tmp))

    # 缺失值处理 --> 滤波处理(去噪）
    signal_data = fill_missing_values(np.ravel(data))  # 处理缺失值
    signal_data = bandpass_filter(signal_data)  # 滤波
    data = signal_data.reshape(data.shape[0], data.shape[1])
    return data, label

# 使用多进程计算原始数据的特征,并进行标准化处理
def compute_features(data):
    #cpu核心数-3
    num_process = multiprocessing.cpu_count() - 3

    # 使用进程池
    with multiprocessing.Pool(processes=num_process) as pool:
        features = pool.map(getFeature, data)
    features = np.array(features)
    print(features.shape)

    # 对特征进行标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    return X


# 交叉验证函数
def cross_validate_lstm(X, y):
    # 超参数范围
    param_grid = {
        'units': [32, 50, 64],  # LSTM单元数量
        'learning_rate': [0.001, 0.01, 0.05],  # 学习率
    }

    # 使用 KFold 进行交叉验证
    kf = KFold(n_splits=CROSS_VALIDATE_NUM, shuffle=True, random_state=42)

    # 记录结果
    best_params = None
    best_score = 0
    max_accuracy = []      #存放准确率

    fold = 1
    for train_index, val_index in kf.split(X):
        print(f"\n===== 训练第 {fold} 折 =====")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        for units in param_grid['units']:
            for learning_rate in param_grid['learning_rate']:
                print(f"尝试参数：units={units}, learning_rate={learning_rate}")
                model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]),
                                          units=units, learning_rate=learning_rate)
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=128,
                    callbacks=[early_stopping],
                    verbose=0
                )

                # 验证集评估
                val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
                print(f"训练集准确率: {val_accuracy:.4f}")

                #取每则交叉验证的最好的准确率作为该则的准确率
                if fold>len(max_accuracy):
                    max_accuracy.append(val_accuracy)
                elif val_accuracy>max_accuracy[-1]:
                    max_accuracy[-1]=val_accuracy

                # 更新最佳参数
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = {'units': units, 'learning_rate': learning_rate}
        fold += 1

    print("\n交叉验证结束")
    print(f"最佳参数：{best_params}")
    print(f"最佳交叉验证准确率：{best_score:.4f}")
    print(f"平均交叉验证准确率：{np.average(np.array(max_accuracy))}")
    return best_params


# 使用模型对输入参数进行预测
# X：测试集上经过原始数据
# 返回：带标签的一维数组
def model_predict(model, X_val):
    # 计算训练集上的特征
    X_val = compute_features(X_val)

    #处理数据格式，使其符合lstm输入要求
    X_val = X_val.reshape(X_val.shape[0], LSTM_TIME_STEPS, int(X_val.shape[1] / LSTM_TIME_STEPS))  # 设置时间步长为3

    y_pred = np.argmax(model.predict(X_val), axis=1)

    return y_pred


# 组装文件的绝对路径
def DataPath(fileNum):
    return DATA_PATH + f"/subject{str(fileNum)}.mat"


def LabelPath(fileNum):
    return LABEL_PATH + f"/{str(fileNum)}/{str(fileNum)}_1.txt"


if __name__ == "__main__":
    # 按照人来划分训练集和测试集
    train_index, val_index = train_test_split([i for i in range(1, SUBJECT_NUM+1)], test_size=0.2, random_state=42)

    # 加载数据并进行预处理
    # train_data, train_label = load_data(train_index)
    val_data, val_label = load_data(val_index)

    # 计算训练集上的特征，给出特征标准化后的数据
    X = compute_features(train_data)

    # 转换X的数据格式，使其符合LSTM的数据规范
    X = X.reshape(X.shape[0], LSTM_TIME_STEPS, int(X.shape[1] / LSTM_TIME_STEPS))  # 设置时间步长为3
    y = train_label

    # 交叉验证获取训练集上最优模型参数
    best_params = cross_validate_lstm(X, y)

    # 使用最佳参数训练最终模型
    print("\n使用最佳参数训练最终模型...")
    # 基于最优参数训练模型
    final_model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]),
                                    units=best_params['units'],
                                    learning_rate=best_params['learning_rate'])
    final_model.fit(X, y, epochs=50, batch_size=128, verbose=1)
    final_model.save("final_model_features8_100.h5")             #保存最终模型

    # 对测试集上的数据进行预测
    y_pred = model_predict(final_model, val_data)

    print(f"模型在测试集上的准确率为：{accuracy_score(val_label, y_pred)}")
