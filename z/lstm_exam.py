# 5个挑选的节点对
from __future__ import division
import pandas as pd
import csv
import numpy as np
import time
import datetime
from collections import defaultdict
import networkx as nx
import scipy
import os


import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, Layer
from keras.layers import Concatenate, Reshape
import keras
import tensorflow as tf


from graph_attention_layer import GraphAttention
from utils import load_data, preprocess_features

from sklearn.preprocessing import MinMaxScaler

import attention

scaler = MinMaxScaler(feature_range=(0, 1))

name = open('../data/name_node_pairs_2_quchong_with12_without_notran.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']


#====================================建图================================
Graph_all = np.load('../data/GAT_preprocess/Graph_all.npy', allow_pickle=True)
temp = np.zeros((739, 739))
A_all = [temp for i in range(5)]
for i in range(5):
    # A_all[i]
    # Graph_all[i]
    for j in range(739):
        if Graph_all[i][j]:

            for k in range(len(Graph_all[i][j])):
                # 第i个时间段，第j个节点对，
                A_all[i][j][Graph_all[i][j][k]] = 1
        else:
            # 若是第i个时间段，第j个节点对，如果该节点在该时间段没有邻接的节点，则跳过，因为初始化就是零
            continue

    A_all[i] = scipy.sparse.csr_matrix(A_all[i])
    A_all[i] = A_all[i] + np.eye(A_all[i].shape[0])

data = []
for i in range(5):
    print(i)
    temp = np.load('../data/GAT_preprocess/'+str(i)+'/all.npy')
    print(temp)
    temp = scipy.sparse.csr_matrix(temp)
    # print(temp)
    temp = temp.tolil()
    data.append(temp)

#=============================超参数========================================
N = data[0].shape[0]                # Number of nodes in the graph
F = data[0].shape[1]                # Original feature dimension

# n_classes = Y_train.shape[1]  # Number of classes
F_ = F                        # Output size of first GraphAttention layer
n_attn_heads = 8              # Number of attention heads in first GAT layer
dropout_rate = 0.6            # Dropout rate (between and inside GAT layers)
l2_reg = 5e-4/2               # Factor for l2 regularization
learning_rate = 5e-3          # Learning rate for Adam
epochs = 10000                # Number of training epochs
es_patience = 100             # Patience fot early stopping

#==================================归一化===================================

data_pre = []
data_train = []
data_test = []
# Preprocessing operations
A_list = A_all


for i in range(5):
    data_pre.append(preprocess_features(data[i]))

for i in range(2):
    data_train.append(data_pre[i])

for i in range(2,4):
    data_test.append(data_pre[i])

# =====================提取交易额作为ytrain==============================

dataset_total = []
for i in range(len(name_node_pairs)):
    file = open('../data/temporal link features_5_7days_739/' +
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
dataset_total = np.array(dataset_total)
dataset_total = np.reshape(dataset_total, (N, 5))

y_train = []
for i in range(2): # 2 train samples
    temp = [a[1+i] for a in dataset_total]
    y_train.append(temp)
y_train = np.array(y_train)
# y_train = scaler.fit_transform(y_train)
print(y_train.shape) #(2, 12)
#===================真实交易额====================================

transum = []
for i in range(2): # 2test samples
    temp =  [a[3+i] for a in dataset_total]
    transum.append(temp)
transum = np.array(transum)
print(transum)
scaler.fit_transform(transum)

#=====================模型构建========================================

X_in = Input(shape=(F,), batch_shape=(N, F))
A_in = Input(shape=(N,), batch_shape=(N, N))
dropout = Dropout(dropout_rate)(X_in)

graph_attention_1 = GraphAttention(N=N,
                                   F_=F_,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([X_in, A_in])
print('graph_attention_1')
print(graph_attention_1)
# graph_attention_1 = tf.reshape(graph_attention_1, [-1,N,F])
graph_attention_1 = Reshape((-1, F))(graph_attention_1)
print('--------graph_attention_1-------------')
print(graph_attention_1)

gat_lstm = LSTM(units=50, input_shape=[N, F], return_sequences=True)(graph_attention_1)
# ---------------------------------GAT_LSTM_att----------------------------------
# 需要将前面的LSTM的returnsequence设为true
gat_lstm = attention.attention_3d_block(gat_lstm)
gat_lstm = Dense(1)(gat_lstm)
# ----------------------------------------------------------------------------------------------------------

model = Model(inputs=[X_in, A_in], outputs=gat_lstm)
model.compile(loss='mse', optimizer='rmsprop')


#=======================拟合模型,预测==================================
# print('data_train')
# print(data_train)
print('y_train')
print(y_train)

print('model')
print(model.summary())


for i in range(2):
    model.fit([data_train[i], A_list[i]], y_train[i], batch_size=N)


value = []
for i in range(2):
    value.append(model.predict([data_test[i], A_list[i+3]], batch_size=N))

value = np.array(value).reshape(2,739)
print(np.array(value).reshape(2,739))
value = scaler.inverse_transform(value)
# 出现负值置零
value[value<0]=0
print(value)

data_pred = pd.DataFrame({'pred_1': value[0], 'pred_2': value[1], 'transum_1': transum[0].tolist(), 'transum_2': transum[1].tolist()})

# -----------------------GAT_LSTM----------------------------------------------------------------

# data_pred.to_csv('./data/temporal link prediction_GAT_LSTM/prediciton_GAT_LSTM.csv', index=False)

# -------------------------GAT_LSTM_GAT-------------------------------------------------------

data_pred.to_csv('../data/temporal link prediction_GAT_LSTM_att/prediciton_GAT_LSTM_att.csv', index=False)

# ---------------------------------------------------------------------------------------------
rmse = 0
rmse = np.sqrt(np.mean(np.power((value - transum), 2)))
print(rmse)
