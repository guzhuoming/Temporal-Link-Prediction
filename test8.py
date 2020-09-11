# 为5个时刻的739个节点对建图，print一下储存的graph文件和邻接矩阵文件
import pandas as pd
import csv
import numpy as np
import time
import datetime
from collections import defaultdict
import networkx as nx

# name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\name_node_pairs_2_quchong_with12_without_notran.csv')
# df_name_node_pairs = pd.read_csv(name)
# df_name_node_pairs.drop(['Unnamed: 0'], axis=1, inplace=True)
# df_name_node_pairs.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\name_node_pairs_2_quchong_with12_without_notran.csv', index=False)

#====================打开文件============================

name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\name_node_pairs_2_quchong_with12_without_notran.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

Graph_all = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\GAT_preprocess\\Graph_all.npy')
A_all = np.load('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\GAT_preprocess\\A_all.npy')

print(Graph_all)
print(A_all)
#==================================================================

