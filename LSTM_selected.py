# 对交易次数多于10次的节点对，取15天为时间间隔，并且从每个节点对的第一笔交易开始连续取10个时间段
# 对这些节点对进行LSTM

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
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

name = open('E:\\0xea674fdde714fd979de3edf0f56aa9716b898ec8\\name_node_pairs_selected.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

mse = 0

for i in range(len(name_node_pairs)):

    file = open('E:\\0xea674fdde714fd979de3edf0f56aa9716b898ec8\\temporal link prediction_1day_5nodepairs_LSTM\\'+name_node_pairs[i]+'.csv','w',newline='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['t', 'tran_sum_real', 'prediction_LSTM', 'difference_LSTM'])
    for j in range(2):
        csvwriter.writerow([j+8, 0., 0., 0.])
    file.close()

# 读取每个节点对的交易记录
for i in range(len(name_node_pairs)):
    print(i)
    print(name_node_pairs[i])

    file = open('E:\\0xea674fdde714fd979de3edf0f56aa9716b898ec8\\temporal link features_5_node_pairs\\' + name_node_pairs[i] + '.csv')
    df = pd.read_csv(file)

    new_data = pd.DataFrame(df, columns=['tran_sum'])
    dataset = new_data.values

    # plt.figure
    # plt.plot(new_data,'x--')
    # plt.show()

    # creating train and test sets
    train = dataset[0:8]
    valid = dataset[8:10]

    # 将dataset转换为x_train, y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []

    for j in range(4, len(train)):
        x_train.append(scaled_data[j-4:j, 0])
        y_train.append(scaled_data[j, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    # predicting values, using past 4 from the train data
    inputs = new_data[len(new_data) - len(valid) - 4:].values

    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    x_test = []
    for j in range(4, inputs.shape[0]):
        x_test.append(inputs[j-4:j, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    tran_sum = model.predict(x_test)
    tran_sum = scaler.inverse_transform(tran_sum)

    rms = np.sqrt(np.mean(np.power((valid-tran_sum), 2)))
    print(rms)
    mse = mse+rms

    file2 = open('E:\\0xea674fdde714fd979de3edf0f56aa9716b898ec8\\temporal link prediction_1day_5nodepairs_LSTM\\' + name_node_pairs[i] + '.csv')
    df_2 = pd.read_csv(file2)
    for j in range(2):
        df_2['prediction_LSTM'][j] = tran_sum[j][0]
        df_2['tran_sum_real'][j] = df['tran_sum'][j+8]
        df_2['difference_LSTM'][j] = df_2['prediction_LSTM'][j]-df_2['tran_sum_real'][j]
    df_2.to_csv('E:\\0xea674fdde714fd979de3edf0f56aa9716b898ec8\\temporal link prediction_1day_5nodepairs_LSTM\\' + name_node_pairs[i] + '.csv', index=False)

    train = new_data[:8]
    valid = new_data[8:]
    valid['Predictions'] = tran_sum
    plt.plot(train['tran_sum'],'x--')
    plt.plot(valid[['tran_sum', 'Predictions']],'x--')
    plt.legend(['tran_sum', 'tran_sum', 'Predictions'])
    plt.xlabel('time')
    plt.ylabel('transaction value')
    plt.show()

mse = mse/len(name_node_pairs)
print(mse)