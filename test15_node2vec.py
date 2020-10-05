from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import math


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, RepeatVector, Reshape, Bidirectional, Input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# 读取节点对-------------------------------------------------------------------------------------------------------------

name = open('./data/name_node_pairs_2_quchong_with12_without_notran.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

# ------------------------------------

# 读取node2vec的txt文件---------------------------

def ReadTxtName(rootdir): #输入l为[]
    f = open(rootdir)
    line = f.readline()
    data_array = []
    while line:
        num = list(map(float, line.split(' ')))
        data_array.append(num)
        line = f.readline()
    f.close()
    # 12 8
    # 0 x x x x x x x x
    # 1 x x x x x x x x
    # 2 ......
    del data_array[0] #删除第一行
    # data_array = list(map(lambda x:x[1:], data_array)) # 删除第一列
    n = 0 #为了后面添加上没有出现的节点的node2vec向量为全零向量
    data_array = sorted(data_array, key=(lambda x: x[0])) #因为node2vec的向量并没有按照节点的顺序排序，我们需要按照每个向量的节点序号排序
    while len(data_array)<739:
        # print(data_array[n][0])
        try:
            # print('try')
            if data_array[n][0] == n:
                n += 1
                continue
            elif data_array[n][0] > n:
                data_array.insert(n, [n] + [0 for i in range(8)])
                n += 1
        except:
            # print('ex')
            # print(len(data_array))
            data_array.append([len(data_array)] + [0 for i in range(8)])

    data_array = list(map(lambda x: x[1:], data_array))  # 删除第一列
    return data_array

data = []
for i in range(5):
    l = ReadTxtName('./data/node2vec/temp_pred_output'+str(i)+'.txt')
    # print(len(l))
    # print(l[0])
    data.append(np.array(l))

# ----------------------------------------------------------
# -------------------------------------------------------------
data_train = []
data_test = []
# print('data.shape[0]')
# print(data.shape[0]) 5
for i in range(2):
    data_train.append(data[i])
for i in range(2,4):
    data_test.append(data[i])
# data_test = np.array(data_test)
# data_train = np.array(data_train)
# 建立model--------------------------------------------------------------------

# input = Input(shape=(8,), batch_shape=(739, 8))
# model = Model(input, Dense(1)(input))
# model.summary()

model = Sequential()
model.add(Dense(1, input_shape=(8,), batch_size=739))
model.summary()
# model.add(RepeatVector(12))
# model.add(Reshape((-1, 1)))
model.compile(loss='mse', optimizer='rmsprop')


# 提取交易额作为ytrain--------------------------------------------------

dataset_total = []
for i in range(len(name_node_pairs)):
    file = open('./data/temporal link features_5_7days_739/' +
                name_node_pairs[i] + '_temp_link_ft.csv')
    df = pd.read_csv(file)
    new_data = pd.DataFrame(df, columns=['tran_sum'])
    dataset = new_data.values # 得到nparray
    dataset = dataset.reshape(1, -1)
    # print('dataset shape')
    # print(dataset.shape) (1, 5)
    dataset_total.append(dataset)
dataset_total = np.array(dataset_total)
scaler = MinMaxScaler(feature_range=(0, 1))

# y_train = dataset_total[:][0:3]
# temp = [] #将12*3的ytrain转化为3*12
# for i in range(3):
#     temp.append(y_train[:][i])
# y_train = temp
# y_train = np.array(y_train)
# print(y_train)

# y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
#----------------------------------------------------------------
# print(data_train.shape)  (3, 12, 8)
dataset_total = np.array(dataset_total)
# print(dataset_total.shape) # (12, 1, 5)
dataset_total = np.reshape(dataset_total, (739, 5))
# print('dataset_total[:][3]')
# print(dataset_total.shape)
# y_train = [a[1:4] for a in dataset_total]
# y_train = np.array(y_train)
y_train = []
for i in range(2): # 3 train samples
    temp = [a[1+i] for a in dataset_total]
    y_train.append(temp)
# y_train = np.array(y_train)
# y_train = scaler.fit_transform(y_train)
print('--------------y_train')
y_train = np.reshape(y_train, (2, 739))
y_train = y_train.tolist()
# print(y_train.shape) #(2, 739)


#---------真实交易额

transum = []
for i in range(2): # 2test samples
    temp =  [a[3+i] for a in dataset_total]
    transum.append(temp)
transum = np.array(transum)
scaler.fit_transform(transum)
print(transum)

#--------------
#----------拟合模型，预测----------
print('==================================data+train==========')
print(data_train[0].shape)

print(data_train[0])
value =[]
for i in range(len(data_train)):
    model.fit(data_train[i], y_train[i], batch_size=739)
    value.append(model.predict(data_test[i], batch_size=739))
print('-----value')
for i in range(len(value)):
    value[i] = np.reshape(value[i], (739))
# value = scaler.inverse_transform(value)
# 出现负值置零
value = np.array(value)
value[value<0]=0
print(value)
data_pred = pd.DataFrame({'pred_1': value[0], 'pred_2': value[1], 'transum_1': transum[0].tolist(), 'transum_2': transum[1].tolist()})
data_pred.to_csv('./data/temporal link prediction_node2vec/prediciton_node2vec.csv', index=False)

#-------------------
rmse = 0
rmse = np.sqrt(np.mean(np.power((value - transum),2)))
print(rmse)
# np.sqrt(np.mean(np.power((valid-tran_sum), 2)))
# print('datareal')
# print(dataset_total)
# print(data)
# print(type(data))
# print(len(data))
# print(len(data[0]))
# print(len(data[0][0]))