# 对LSTM预测出负值的置0
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

name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\name_node_pairs_2_quchong_with12_without_notran.csv')
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

for i in range(len(name_node_pairs)):
    print(i)
    print(str(i / len(name_node_pairs) * 100) + '%')

    file2 = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\temporal link prediction_LSTM\\' + name_node_pairs[i] + '.csv')
    df_2 = pd.read_csv(file2)
    for j in range(len(df_2['prediction_LSTM'])):
        if df_2['prediction_LSTM'][j]<0:
            df_2['prediction_LSTM'][j] = 0
    df_2.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\temporal link prediction_LSTM_withoutNegative\\' + name_node_pairs[i] + '.csv', index=False)
    # print(df_2['tran_sum_real'].values)
    # print(df_2['prediction_LSTM'].values)
    # print(df_2['tran_sum_real'].values - df_2['prediction_LSTM'].values)
    rms = (np.mean(np.power((df_2['tran_sum_real'].values - df_2['prediction_LSTM'].values), 2)))
    # print(rms)
    mse = mse+rms
    print(mse)

    # train = new_data[:3]
    # valid = new_data[3:]
    # valid['Predictions'] = tran_sum
    # plt.plot(train['tran_sum'],'x--')
    # plt.plot(valid[['tran_sum', 'Predictions']],'x--')
    # plt.legend(['tran_sum', 'tran_sum', 'Predictions'])
    # plt.xlabel('time')
    # plt.ylabel('transaction value')
    # plt.show()

rmse = np.sqrt(mse/len(name_node_pairs))
print(rmse)