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
scaler = MinMaxScaler(feature_range=(0, 1))

name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\name_node_pairs_selected_0x3f_5_7days.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

#====================================建图================================
# 遍历12个节点对，记录邻接关系
setlist = [] # 储存12个节点对的名字，每个节点对用集合来储存，两个节点的排序没影响，其实12个节点对都是有邻接关系的，因为都是同一个源节点爬取的一阶节点
for j in range(len(name_node_pairs)):
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link features_5_7days\\'+name_node_pairs[j]+'_temp_link_ft.csv')
    df = pd.read_csv(file)
    tempset = set()
    s = name_node_pairs[j]
    index_ = s.find('_')
    a = s[0:index_]
    b = s[index_+1:]
    tempset.add(a)
    tempset.add(b)
    setlist.append(tempset)

graph = defaultdict(list)
# 为了运用openne的node2vec做的egdelist
txtName = "E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\edgelist.txt"
f = open(txtName, "a+")
# 在下面的循环中做edgelist
for i in range(len(name_node_pairs)):
    for j in range(len(name_node_pairs)):
        if i!=j:
            if setlist[i]&setlist[j]:
                graph[i].append(j)
                new_context = str(i)+' '+str(j)+'\n'
                f.write(new_context)
        else:
            continue
f.close()



# Read data
# X = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(0)+'\\all.npy')
# print(X)
# X_out = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(3)+'\\all.npy')
# X = scipy.sparse.csr_matrix(X)
# X = X.tolil()
# X_out = scipy.sparse.csr_matrix(X_out)
# X_out = X_out.tolil()

data = []
for i in range(5):
    print(i)
    temp = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(i)+'\\all.npy')
    print(temp)
    temp = scipy.sparse.csr_matrix(temp)
    # print(temp)
    temp = temp.tolil()
    data.append(temp)
# ----------------------------
# data_train = []
# data_test = []
# # print('data.shape[0]')
# # print(data.shape[0]) 5
# for i in range(2):
#     data_train.append(data[i])
# for i in range(2,4):
#     data_test.append(data[i])
# data_test = np.array(data_test)
# data_train = np.array(data_train)


# X1 = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(0)+'\\all.npy')
# print(X1)
# # X1_out = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(3)+'\\all.npy')
# X1 = scipy.sparse.csr_matrix(X1)
# # X1 = X1.tolil()
# # X1_out = scipy.sparse.csr_matrix(X1_out)
# # X1_out = X1_out.tolil()
#
# X2 = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(1)+'\\all.npy')
# print(X2)
# # X2_out = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(3)+'\\all.npy')
# X2 = scipy.sparse.csr_matrix(X2)
# # X2 = X2.tolil()
# # X2_out = scipy.sparse.csr_matrix(X2_out)
# # X2_out = X2_out.tolil()
#
# X3 = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(2)+'\\all.npy')
# print(X3)
# # X3_out = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(3)+'\\all.npy')
# X3 = scipy.sparse.csr_matrix(X3)
# # X3 = X3.tolil()
# # X3_out = scipy.sparse.csr_matrix(X3_out)
# # X3_out = X3_out.tolil()
#
# X4 = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(3)+'\\all.npy')
# print(X4)
# # X4_out = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(3)+'\\all.npy')
# X4 = scipy.sparse.csr_matrix(X4)
# # X4 = X4.tolil()
# # X4_out = scipy.sparse.csr_matrix(X4_out)
# # X4_out = X4_out.tolil()
#
# X5 = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(4)+'\\all.npy')
# print(X5)
# # X5_out = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\'+str(3)+'\\all.npy')
# X5 = scipy.sparse.csr_matrix(X5)
# # X5 = X5.tolil()
# # X5_out = scipy.sparse.csr_matrix(X5_out)
# # X5_out = X5_out.tolil()

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
# print('A')
# print(A)

# ===================    A    ===============================

A = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
# A = A.A
# A = A.tolist()
# print('A--------------')
# print(A)
#
# A = [A for i in range(2)]
# A = np.array(A)
# A = A.reshape(2*N,N)
#
# A = scipy.sparse.csr.csr_matrix(A)

# print('A')
# print(type(A))
# print(A)
A = A + np.eye(A.shape[0])  # Add self-loops
# print('A = A+np.eye(A.shape[0])')
# print(type(A))
# print(A)

#=========================  X      =======================
#==================================归一化===================================

data_pre = []
data_train = []
data_test = []
# Preprocessing operations
A_list = [A for i in range(5)]

for i in range(5):
    data_pre.append(preprocess_features(data[i]))

for i in range(2):
    data_train.append(data_pre[i])

for i in range(2,4):
    data_test.append(data_pre[i])

# for i in range(5):
#     data_pre.append(preprocess_features(data[i]))
# for i in range(3):
#     data_train.append([data_pre[i], A])
# for i in range(2,4):
#     data_test.append([data_pre[i], A])

#--ValueError: could not broadcast input array from shape (12,7) into shape (12)
# for i in range(5):
#     data_pre.append(preprocess_features(data[i]))
# for i in range(3):
#     data_train.append(np.array([np.array(data_pre[i]).ravel(), np.array(A).ravel()]))
# for i in range(2,4):
#     data_test.append([data_pre[i], A])
#=====================提取交易额作为ytrain==============================

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


# X1 = preprocess_features(X1)
# X2 = preprocess_features(X2)
# X3 = preprocess_features(X3)
# X4 = preprocess_features(X4)
# X5 = preprocess_features(X5)

#=====================模型构建========================================



# Model definition (as per Section 3.3 of the paper)
# X1_in = Input(shape=(F,))
# X2_in = Input(shape=(F,))
# X3_in = Input(shape=(F,))
# X4_in = Input(shape=(F,))
# X5_in = Input(shape=(F,))
X_in = Input(shape=(F,), batch_shape=(N, F))
A_in = Input(shape=(N,), batch_shape=(N, N))
dropout = Dropout(dropout_rate)(X_in)

# dropout1 = Dropout(dropout_rate)(X1_in)
# dropout2 = Dropout(dropout_rate)(X2_in)
# dropout3 = Dropout(dropout_rate)(X3_in)
# dropout4 = Dropout(dropout_rate)(X4_in)
# dropout5 = Dropout(dropout_rate)(X5_in)

graph_attention_1 = GraphAttention(N=N,
                                   F_=F_,
                                   attn_heads=n_attn_heads,
                                   attn_heads_reduction='average',
                                   dropout_rate=dropout_rate,
                                   activation='elu',
                                   kernel_regularizer=l2(l2_reg),
                                   attn_kernel_regularizer=l2(l2_reg))([X_in, A_in])
print(graph_attention_1)
# graph_attention_1 = tf.reshape(graph_attention_1, [-1,N,F])
graph_attention_1 = Reshape((-1, F))(graph_attention_1)
print('--------graph_attention_1-------------')
print(graph_attention_1)
# graph_attention_1 = RepeatVector(N)(graph_attention_1)
gat_lstm = LSTM(units=50, input_shape=[N, F], return_sequences=False)(graph_attention_1)
# gat_lstm_gat = GraphAttention(N=N,
#                           F_=F_,
#                           attn_heads=n_attn_heads,
#                           attn_heads_reduction='average',
#                           dropout_rate=dropout_rate,
#                           activation='elu',
#                           kernel_regularizer=l2(l2_reg),
#                           attn_kernel_regularizer=l2(l2_reg))([gat_lstm, A_in])
# gat_lstm = Dense(1)(gat_lstm_gat)
gat_lstm = Dense(1)(gat_lstm)

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
    model.fit([data_train[i], A_list[i]], y_train[i], batch_size=12)

value = []
for i in range(2):
    value.append(model.predict([data_test[i], A_list[i+3]]))

value = np.array(value).reshape(2,12)
print(np.array(value).reshape(2,12))
value = scaler.inverse_transform(value)
# 出现负值置零
value[value<0]=0
print(value)
rmse = 0
rmse = np.sqrt(np.mean(np.power((value - transum),2)))
print(rmse)
#==========================后面先不管=============================================
# model = Sequential
# model.add(graph_attention_1)
# model.add(LSTM(units=50, input_shape=[N,F], return_sequences=False))
# model.add(Dense(12))
# model.compile(loss='mse', optimizer='rmsprop')


# graph_attention_2 = GraphAttention(F_,
#                                    attn_heads=n_attn_heads,
#                                    attn_heads_reduction='average',
#                                    dropout_rate=dropout_rate,
#                                    activation='elu',
#                                    kernel_regularizer=l2(l2_reg),
#                                    attn_kernel_regularizer=l2(l2_reg))([X_in, A_in])
# print(graph_attention_2)

# graph_attention_3 = GraphAttention(F_,
#                                    attn_heads=n_attn_heads,
#                                    attn_heads_reduction='average',
#                                    dropout_rate=dropout_rate,
#                                    activation='elu',
#                                    kernel_regularizer=l2(l2_reg),
#                                    attn_kernel_regularizer=l2(l2_reg))([X_in, A_in])
# print(graph_attention_3)

# graph_attention_4 = GraphAttention(F_,
#                                    attn_heads=n_attn_heads,
#                                    attn_heads_reduction='average',
#                                    dropout_rate=dropout_rate,
#                                    activation='elu',
#                                    kernel_regularizer=l2(l2_reg),
#                                    attn_kernel_regularizer=l2(l2_reg))([X_in, A_in])
# print(graph_attention_4)

# graph_attention_5 = GraphAttention(F_,
#                                    attn_heads=n_attn_heads,
#                                    attn_heads_reduction='average',
#                                    dropout_rate=dropout_rate,
#                                    activation='elu',
#                                    kernel_regularizer=l2(l2_reg),
#                                    attn_kernel_regularizer=l2(l2_reg))([X_in, A_in])
# print(graph_attention_5)

# Build model
# optimizer = Adam(lr=learning_rate)
# model = Model(inputs=[X1_in, A_in], outputs=graph_attention_1)
# model.compile(optimizer=optimizer,
#               loss='mean_squared_error',
#               weighted_metrics=['acc'])
# print(graph_attention_1)

# model = Model(inputs=[X_in, A_in], outputs=graph_attention_1)
#
# optimizer = Adam(lr=learning_rate)
# model.compile(optimizer=optimizer,
#               loss='mean_squared_error',
#               weighted_metrics=['acc'])
# model.summary()

# # Callbacks
# es_callback = EarlyStopping(monitor='val_weighted_acc', patience=es_patience)
# tb_callback = TensorBoard(batch_size=N)
# mc_callback = ModelCheckpoint('logs/best_model.h5',
#                               monitor='val_weighted_acc',
#                               save_best_only=True,
#                               save_weights_only=True)

# Train model
# validation_data = ([X, A], Y_val, idx_val)
# model.fit([X, A],
#           Y_train,
#           sample_weight=idx_train,
#           epochs=epochs,
#           batch_size=N,
#           validation_data=validation_data,
#           shuffle=False,  # Shuffling data means shuffling the whole graph
#           callbacks=[es_callback, tb_callback, mc_callback])
#
# # Load best model
# model.load_weights('logs/best_model.h5')
#
# # Evaluate model
# eval_results = model.evaluate([X, A],
#                               Y_test,
#                               sample_weight=idx_test,
#                               batch_size=N,
#                               verbose=0)
# print('Done.\n'
#       'Test loss: {}\n'
#       'Test accuracy: {}'.format(*eval_results))
