from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Input, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, Layer
from keras.layers import Concatenate, Reshape
import keras

from graph import GraphConvolution
from utils import *
from graph_attention_layer import GraphAttention
import scipy
import pandas as pd

import time

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

name = open('./data/name_node_pairs_2_quchong_with12_without_notran.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

# Define parameters

FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization

# ====================================建图================================
Graph_all = np.load('./data/GAT_preprocess/Graph_all.npy')
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
    # A_all[i] = A_all[i] + np.eye(A_all[i].shape[0])

# Normalize X
# X /= X.sum(1).reshape(-1, 1)
# print('Normalize X')
# print(X)

data = []
for i in range(5):
    print(i)
    temp = np.load('./data/GAT_preprocess/'+str(i)+'/all.npy')
    # print('temp')
    temp = temp.tolist()
    temp = np.mat(temp)
    data.append(temp)

N = data[0].shape[0]                # Number of nodes in the graph
F = data[0].shape[1]                # Original feature dimension
F_ = F                        # Output size of first GraphAttention layer

# GAT---------------------------------------------------------------------------------
n_attn_heads = 8              # Number of attention heads in first GAT layer
dropout_rate = 0.6            # Dropout rate (between and inside GAT layers)
l2_reg = 5e-4/2               # Factor for l2 regularization
learning_rate = 5e-3          # Learning rate for Adam
epochs = 10000                # Number of training epochs
es_patience = 100             # Patience fot early stopping


# =====================提取交易额作为ytrain==============================

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
dataset_total = np.array(dataset_total)
dataset_total = np.reshape(dataset_total, (N, 5))

y_train = []
for i in range(2): # 2 train samples
    temp = [a[1+i] for a in dataset_total]
    y_train.append(temp)
y_train = np.array(y_train)
# y_train = scaler.fit_transform(y_train)
print(y_train.shape) #(2, 12)

# ------------------------------------------------------------

#===================真实交易额====================================

transum = []
for i in range(2): # 2test samples
    temp =  [a[3+i] for a in dataset_total]
    transum.append(temp)
transum = np.array(transum)
print(transum)
scaler.fit_transform(transum)

# =======================================================================
A_list = A_all
if FILTER == 'localpool':
    A_list_ = []
    graph_all = []
    for i in range(5):
        A_ = preprocess_adj(A_list[i], SYM_NORM)
        A_list_.append(A_)
        graph = [data[i], A_]
        graph_all.append(graph)
    support = 1
    G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]


# elif FILTER == 'chebyshev':
#     """ Chebyshev polynomial basis filters (Defferard et al., NIPS 2016)  """
#     print('Using Chebyshev polynomial basis filters...')
#     L = normalized_laplacian(A, SYM_NORM)
#     L_scaled = rescale_laplacian(L)
#     T_k = chebyshev_polynomial(L_scaled, MAX_DEGREE)
#     support = MAX_DEGREE + 1
#     graph = [X]+T_k
#     G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True) for _ in range(support)]

else:
    raise Exception('Invalid filter type.')

# --------------------------------------------------------

data_train = []
data_test = []

for i in range(2):
    data_train.append(graph_all[i])

for i in range(2,4):
    data_test.append(graph_all[i])


X_in = Input(shape=(F,), batch_shape=(N, F))

# Define model architecture
# NOTE: We pass arguments for graph convolutional layers as a list of tensors.
# This is somewhat hacky, more elegant options would require rewriting the Layer base class.
# H = Dropout(0.5)(X_in)

H = GraphConvolution(F_, support, activation='relu', kernel_regularizer=l2(5e-4))([X_in]+G)
H = Reshape((-1, F))(H)
gcn_lstm = LSTM(units=50, input_shape=[N, F], return_sequences=False)(H)
print('gcn_lstm')
print(gcn_lstm)
# ------------如果是gcn_lstm-------------------------

gcn_lstm = Dense(1)(gcn_lstm)


# ------------------------如果是gcn_lstm_gat---------------------------------

# A_in = Input(shape=(N,), batch_shape=(N, N))
# gcn_lstm_gat = GraphAttention(N=N,
#                               F_=F_,
#                               attn_heads=n_attn_heads,
#                               attn_heads_reduction='average',
#                               dropout_rate=dropout_rate,
#                               activation='elu',
#                               kernel_regularizer=l2(l2_reg),
#                               attn_kernel_regularizer=l2(l2_reg))([gcn_lstm, A_in])
# gcn_lstm = Dense(1)(gcn_lstm_gat)

# ---------------------------------------------------------

# Compile model
model = Model(inputs=[X_in]+G, outputs=gcn_lstm)
# model = Model(inputs=[[X_in]+G, A_in], outputs=gcn_lstm)
model.compile(loss='mse', optimizer='rmsprop') # 'rmsprop'

#=======================拟合模型,预测==================================
# print('data_train')
# print(data_train)
print('y_train')
print(y_train)

print('model')
print(model.summary())


for i in range(2):
    # model.fit([data_train[i], A_list[i]], y_train[i], batch_size=N)
    model.fit(data_train[i], y_train[i], batch_size=N)

value = []
for i in range(2):
    # value.append(model.predict([data_test[i], A_list[i+3]], batch_size=N))
    value.append(model.predict(data_test[i], batch_size=N))

value = np.array(value).reshape(2,739)
print(np.array(value).reshape(2,739))
# value = scaler.inverse_transform(value)
# 出现负值置零
value[value<0]=0
print(value)
data_pred = pd.DataFrame({'pred_1': value[0], 'pred_2': value[1], 'transum_1': transum[0].tolist(), 'transum_2': transum[1].tolist()})
data_pred.to_csv('./data/temporal link prediction_GCN_LSTM/prediciton_GCN_LSTM.csv', index=False)


rmse = 0
rmse = np.sqrt(np.mean(np.power((value - transum),2)))
print(rmse)