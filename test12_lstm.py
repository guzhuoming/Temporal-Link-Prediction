# 对交易次数多于10次的节点对，取15天为时间间隔，并且从每个节点对的第一笔交易开始连续取10个时间段
# 对这些节点对进行LSTM

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import math

# setting figure size
from matplotlib.pylab import rcParams

# for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

name = open('./data/name_node_pairs_2_quchong_with12_without_notran.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

mse = 0

# for i in range(len(name_node_pairs)):
#
#     file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\temporal link prediction_LSTM\\'+name_node_pairs[i]+'.csv','w',newline='')
#     csvwriter = csv.writer(file)
#     csvwriter.writerow(['t', 'tran_sum_real', 'prediction_LSTM', 'difference_LSTM'])
#     for j in range(2):
#         csvwriter.writerow([j+3, 0., 0., 0.])
#     file.close()

# 读取每个节点对的交易记录

for i in range(700, len(df_name_node_pairs)):
    print(i)
    print(str(i / len(name_node_pairs) * 100) + '%')

    file = open('./data/temporal link features_5_7days_739/' + name_node_pairs[i] + '_temp_link_ft.csv')
    df = pd.read_csv(file)

    new_data = pd.DataFrame(df, columns=['tran_sum'])
    # print('type(new_data)')
    # print(type(new_data))
    dataset = new_data.values
    # print('type(dataset)')
    # print((type(dataset)))

    # plt.figure
    # plt.plot(new_data,'x--')
    # plt.show()

    # creating train and test sets
    train = dataset[0:3]
    valid = dataset[3:5]

    # 将dataset转换为x_train, y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []

    for j in range(2, len(train)):
        x_train.append(scaled_data[j-2:j, 0])
        y_train.append(scaled_data[j, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #create and fit the LSTM network
    model = Sequential()
    # print('x_train.shape[1]')
    # print(x_train.shape[1])
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # print('xtrian')
    # print(x_train.shape)
    # print('y_train')
    # print(y_train.shape)
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    # predicting values, using past 2 from the train data
    inputs = new_data[len(new_data) - len(valid) - 2:].values

    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    x_test = []
    for j in range(2, inputs.shape[0]):
        # print('inputs.shape[0]')   6
        # print(inputs.shape[0])
        x_test.append(inputs[j-2:j, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    tran_sum = model.predict(x_test)
    tran_sum = scaler.inverse_transform(tran_sum)

    # rms = np.sqrt(np.mean(np.power((valid-tran_sum), 2)))
    rms = (np.mean(np.power((valid - tran_sum), 2)))
    # print(rms)
    mse = mse+rms
    print(mse)

    file2 = open('./data/temporal link prediction_LSTM/' + name_node_pairs[i] + '.csv')
    df_2 = pd.read_csv(file2)
    for j in range(2):
        df_2['prediction_LSTM'][j] = tran_sum[j][0]
        df_2['tran_sum_real'][j] = df['tran_sum'][j+3]
        df_2['difference_LSTM'][j] = df_2['prediction_LSTM'][j]-df_2['tran_sum_real'][j]
    df_2.to_csv('./data/temporal link prediction_LSTM/' + name_node_pairs[i] + '.csv', index=False)

    # train = new_data[:3]
    # valid = new_data[3:]
    # valid['Predictions'] = tran_sum
    # plt.plot(train['tran_sum'],'x--')
    # plt.plot(valid[['tran_sum', 'Predictions']],'x--')
    # plt.legend(['tran_sum', 'tran_sum', 'Predictions'])
    # plt.xlabel('time')
    # plt.ylabel('transaction value')
    # plt.show()

mse = np.sqrt(mse/len(name_node_pairs))
print(mse)