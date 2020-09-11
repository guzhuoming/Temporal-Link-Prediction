from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import math

import seq2seq
from seq2seq.models import SimpleSeq2Seq
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, RepeatVector, Reshape, Bidirectional

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# 读取节点对-------------------------------------------------------------------------------------------------------------

name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\name_node_pairs_selected_0x3f_5_7days.csv')
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
    data_array = list(map(lambda x:x[1:], data_array)) # 删除第一列
    # data_array = np.array(data_array)
    # data_array = np.delete(data_array, 0, axis=0)
    return data_array

data = []
for i in range(5):
    l = ReadTxtName('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\node2vec_12nodepairs\\temp_pred_output'+str(i+1)+'.txt')
    data.append(l)
# ----------------------------------------------------------
# -------------------------------------------------------------
data_train = []
data_test = []
# print('data.shape[0]')
# print(data.shape[0]) 5
for i in range(3):
    data_train.append(data[i])
for i in range(2,4):
    data_test.append(data[i])
data_test = np.array(data_test)
data_train = np.array(data_train)
# 建立model--------------------------------------------------------------------

model = Sequential()
model.add(Bidirectional(LSTM(units=50, input_shape=[ 12, 8], return_sequences=False))) #加不加biderectional好像没影响
# model.add(LSTM(units=50))  # 这一层加不加好像也没影响，如果加，要把第一层的return sequence置为true
model.add(Dense(12))
# model.add(RepeatVector(12))
# model.add(Reshape((-1, 1)))
model.compile(loss='mse', optimizer='rmsprop')


# 提取交易额作为ytrain--------------------------------------------------

dataset_total = []
for i in range(len(name_node_pairs)):
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link features_5_7days\\' +
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
dataset_total = np.reshape(dataset_total, (12, 5))
# print('dataset_total[:][3]')
# print(dataset_total.shape)
# y_train = [a[1:4] for a in dataset_total]
# y_train = np.array(y_train)
y_train = []
for i in range(3): # 3 train samples
    temp = [a[1+i] for a in dataset_total]
    y_train.append(temp)
y_train = np.array(y_train)
y_train = scaler.fit_transform(y_train)
print(y_train.shape) #(3, 12)

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

model.fit(data_train, y_train)
value = model.predict(data_test)
value = scaler.inverse_transform(value)
# 出现负值置零
value[value<0]=0
print(value)

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