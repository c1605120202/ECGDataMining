from tensorflow.keras.models import load_model

final_model=load_model("fianl_model.h5")
y_pred = final_model.predict(X_val)
print(y_val)
print(y_pred)
print(f"模型在测试集上的准确率为：{accuracy_score(y_val, y_pred)}")