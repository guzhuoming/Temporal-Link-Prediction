# 为5个时刻的739个节点对建图
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

Graph_all = []
A_all = []
#=====================建图=================================
for i in range(5):
    print(i)
    setlist = [] # 储存739个节点对的名字，每个节点对用集合来储存
    for j in range(len(name_node_pairs)):
        file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\temporal link features_5_7days\\'+name_node_pairs[j]+'_temp_link_ft.csv')
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
    txtName = "E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\node2vec\\edgelist_"+str(i)+".txt"
    f = open(txtName, "a+")
    # 在下面的循环中做edgelist
    for ii in range(len(name_node_pairs)):
        print('ii')
        print(ii)
        for jj in range(len(name_node_pairs)):
            if ii!=jj:
                if setlist[ii]&setlist[jj]:
                    #不仅要有邻接关系，还要看在那个时间段是否有邻接关系
                    file1 = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\temporal link features_5_7days\\'+ name_node_pairs[ii] +'_temp_link_ft.csv')
                    df1 = pd.read_csv(file1)
                    file2 = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\temporal link features_5_7days\\' + name_node_pairs[jj] + '_temp_link_ft.csv')
                    df2 = pd.read_csv(file2)
                    temp1 = df1['tran_num'][i]
                    temp2 = df2['tran_num'][i]
                    if temp1!=0 and temp2!=0:
                        graph[ii].append(jj)
                        new_context = str(ii)+' '+str(jj)+'\n'
                        f.write(new_context)
            else:
                continue

    f.close()
    Graph_all.append(graph)
    A = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    A_all.append(A)
Graph_all = np.array(Graph_all)
np.save('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\GAT_preprocess\\Graph_all.npy', Graph_all)
A_all = np.array(A_all)
np.save('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\GAT_preprocess\\A_all.npy', A_all)
#==================================================================
# 创建带有时间序号的空的temporal link feature 文件
for i in range(5):
    # 分五个时间段分别储存
    li_all = [] #记录所有节点对的特征
    # 遍历12个节点对，记录特征
    for j in range(len(name_node_pairs)):
        file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\temporal link features_5_7days\\'+name_node_pairs[j]+'_temp_link_ft.csv')
        df = pd.read_csv(file)
        li_all.append([df['tran_num'][i], df['tran_sum'][i], df['tran_mean'][i], df['interval_mean'][i], df['interval_var'][i], df['tran_freq'][i], df['tran_num_dir'][i]])

    a_all = np.array(li_all)
    np.save('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\GAT_preprocess\\' + str(i) + '\\all' + '.npy', a_all)
    # 读取：np.load