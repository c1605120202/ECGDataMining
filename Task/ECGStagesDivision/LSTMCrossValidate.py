import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

CROSS_VALIDATE_NUM = 5  # 表示几则交叉验证


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


# 交叉验证函数
def cross_validate_lstm(X, y):
    # 超参数范围
    param_grid = {
        'units': [32, 50, 64],  # LSTM单元数量
        'learning_rate': [0.01, 0.025, 0.05, 0.075],  # 学习率
    }

    # 使用 KFold 进行交叉验证
    kf = KFold(n_splits=CROSS_VALIDATE_NUM, shuffle=True, random_state=42)

    # 记录结果
    best_params = None
    best_score = 0
    max_accuracy = []  # 存放准确率

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

                # 取每则交叉验证的最好的准确率作为该则的准确率
                if fold > len(max_accuracy):
                    max_accuracy.append(val_accuracy)
                elif val_accuracy > max_accuracy[-1]:
                    max_accuracy[-1] = val_accuracy

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
