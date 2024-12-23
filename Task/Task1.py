import pandas as pd
import numpy as np
import scipy.stats as st

#计算Petrosian分形维数
def petrosian_fd(time_series):
    N = len(time_series)

    # 计算符号变化的次数（零交叉点）
    diff = np.diff(np.sign(time_series))  # 计算符号变化
    N_v = np.sum(diff != 0)  # 符号变化的次数

    # 计算 Petrosian 分形维数
    if N_v == 0:  # 处理特殊情况，避免除以 0
        return 0
    return np.log10(N) / (np.log10(N) + np.log10(N / N_v))

#根据文件存放目录读取文件
data=np.array(np.loadtxt("../DataFile/EEG.txt"))
leap=30*125     #表示每帧中含有的数据量

#将全部的数据进行分组，分成1084*（30*125）
i,result=1,[]
while i*leap<=data.__len__():
    temp=data[(i-1)*leap:i*leap]
    result.append(temp)
    i+=1
#当数据不是整步长时，对最后一节数据进行单独分组
if i*leap-data.__len__()<leap:
    result.append(data[(i-1)*leap:data.__len__()])

#分别表示：均值,标准差，中值，Petrosian分形维数，偏度，峰度
means,stds,medians,petrosians,skewnesses,kurtosises=[],[],[],[],[],[]
for item in result:
    #将参数加入对应的列表中
    means.append(np.average(item))
    stds.append(np.std(item))
    medians.append(np.median(item))
    petrosians.append(petrosian_fd(item))
    skewnesses.append(st.skew(item))
    kurtosises.append(st.kurtosis(item))

#将特征列表合并，生成Excel表格
df=pd.DataFrame({"均值":means,"标准差":stds,"中值":medians,"Petrosian分形维数":petrosians,"偏度":skewnesses,"峰度":kurtosises})
df.to_excel("D://test//result1.xlsx",index=False)
