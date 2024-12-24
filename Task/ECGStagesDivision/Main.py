import scipy.io

from Task.ECGStagesDivision.DataPreprocess import *
from Task.ECGStagesDivision.FeatureExtract import *
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# 定义 LSTM 模型
def create_lstm_model(input_shape):
    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))  # 5类输出，使用 softmax 激活
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#文件根目录
DATA_PATH="X:/Dataset/data"
LABEL_PATH="X:/Dataset/label"

#组装文件的绝对路径
def DataPath(fileNum):
    return DATA_PATH+f"/subject{str(fileNum)}.mat"
def LabelPath(fileNum):
    return LABEL_PATH+f"/{str(fileNum)}/{str(fileNum)}_1.txt"

data,label=np.empty((0,6000)),np.empty(0)
for i in range(1,2):
    print(i)
    # 读取X_1.txt作为标签文件
    with open(LabelPath(i), 'r') as file:
        data_list = file.readlines()
    # 去掉每行末尾的换行符
    data_list = [line.strip() for line in data_list]

    tmp=[]    #标签数据集合
    for k in range(0,len(data_list)):
        if data_list[k]=='':
            continue
        tmp.append(int(data_list[k]))
    tmp=tmp[0:-30] # 舍弃掉后30个epoch的标签
    #标签中跳过了4直接到5，由于需要生成one-hot编码所以需要进行转换
    label=np.concatenate((label,np.array([4 if num==5 else num for num in tmp])))

    # 加载 .mat 文件作为数据文件
    mat_file = scipy.io.loadmat(DataPath(i))
    # 访问某个特定的变量（例如：变量名是 'data'）
    dt=mat_file['X2']
    data = np.concatenate((data,dt),axis=0)

    print(len(dt))
    print(len(tmp))


#缺失值处理 --> 滤波处理(去噪）
signal_data=fill_missing_values(np.ravel(data))       #处理缺失值
signal_data=bandpass_filter(signal_data)         #滤波
data=signal_data.reshape(data.shape[0],data.shape[1])

#特征提取
features=getFeature(data)

#对特征进行标准化
scaler=StandardScaler()

X=scaler.fit_transform(features)
#转换X的数据格式，使其符合LSTM的数据规范
X=X.reshape(X.shape[0],4,int(X.shape[1]/4))         #设置时间步长为4

# 将标签转换为 one-hot 编码
label=label[0:100]
y = to_categorical(label,num_classes=5)

# 使用 KFold 进行交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 五折交叉验证
accuracies = []  # 存储每一折的准确率

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Training fold {fold + 1}...")

    # 获取训练集和验证集
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 创建并训练 LSTM 模型
    model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # 使用 EarlyStopping 防止过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 训练模型
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val),callbacks=[early_stopping], verbose=1)

    # 在验证集上评估模型
    y_pred_prob = model.predict(X_val)  # 输出每个类别的概率
    y_pred = np.argmax(y_pred_prob, axis=1)  # 选择概率最大的类别作为预测结果
    y_val_labels = np.argmax(y_val, axis=1)  # 将 one-hot 编码标签转换为整数标签

    accuracy = accuracy_score(y_val_labels, y_pred)
    accuracies.append(accuracy)

    print(f"Fold {fold + 1} accuracy: {accuracy:.4f}")

# 输出平均准确率
print(f"Average accuracy across 5 folds: {np.mean(accuracies):.4f}")




