import scipy.io

from Task.ECGStagesDivision.DataPreprocess import *
from Task.ECGStagesDivision.Evaluate import *
from Task.ECGStagesDivision.FeatureExtract import *

from sklearn.model_selection import train_test_split

from Task.ECGStagesDivision.LSTMCrossValidate import create_lstm_model, cross_validate_lstm

# 文件根目录
DATA_PATH = "D:/Dataset/data"
LABEL_PATH = "D:/Dataset/label"
SUBJECT_NUM = 100  # 要读取的subject数量
LSTM_TIME_STEPS = 30  # 算法时间步


# 组装文件的绝对路径
def DataPath(fileNum):
    return DATA_PATH + f"/subject{str(fileNum)}.mat"


def LabelPath(fileNum):
    return LABEL_PATH + f"/{str(fileNum)}/{str(fileNum)}_1.txt"


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
    # signal_data = bandpass_filter(signal_data)  # 滤波
    data = signal_data.reshape(data.shape[0], data.shape[1])
    return data, label


if __name__ == "__main__":
    # 按照人来划分训练集和测试集
    train_index, val_index = train_test_split([i for i in range(1, SUBJECT_NUM + 1)], test_size=0.2, random_state=42)

    # 加载数据并进行预处理
    train_data, train_label = load_data(train_index)
    val_data, val_label = load_data(val_index)

    # 计算训练集上的特征，给出特征标准化后的数据
    X = compute_features(train_data)

    # 转换X的数据格式，使其符合LSTM的数据规范
    X = X.reshape(X.shape[0], LSTM_TIME_STEPS, int(X.shape[1] / LSTM_TIME_STEPS))  # 设置时间步长为30
    y = train_label

    # 交叉验证获取训练集上最优模型参数
    best_params = cross_validate_lstm(X, y)

    # 使用最佳参数训练最终模型
    print("\n使用最佳参数训练最终模型...")
    final_model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]),
                                    units=best_params['units'],
                                    learning_rate=best_params['learning_rate'])
    final_model.fit(X, y, epochs=50, batch_size=128, verbose=1)
    final_model.save(f"final_model_features6_{SUBJECT_NUM}.h5")  # 保存最终模型

    # 对最终的模型进行性能评估
    evaluate(final_model, val_data, val_label)
