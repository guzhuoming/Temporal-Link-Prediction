# 5个挑选的节点对
import pandas as pd
import csv
import numpy as np
import time
import datetime
from collections import defaultdict

name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\name_node_pairs_selected_0x3f_5_7days.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

# 创建带有时间序号的空的temporal link feature 文件
for i in range(5):
    # 分五个时间段分别储存
    li_train = [] #记录前九个节点对的特征
    li_test = [] #记录后三个节点对的特征
    li_all = [] #记录所有节点对的特征
    # 遍历12个节点对，记录特征
    for j in range(len(name_node_pairs)):
        file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link features_5_7days\\'+name_node_pairs[j]+'_temp_link_ft.csv')
        df = pd.read_csv(file)
        li_all.append([df['tran_num'][i], df['tran_sum'][i], df['tran_mean'][i], df['interval_mean'][i], df['interval_var'][i], df['tran_freq'][i], df['tran_num_dir'][i]])
        # 划分训练集和测试集，前九个节点对为训练集，后3个节点对为测试集
        if j < 9:
            li_train.append([df['tran_num'][i], df['tran_sum'][i], df['tran_mean'][i], df['interval_mean'][i], df['interval_var'][i], df['tran_freq'][i], df['tran_num_dir'][i]])
        else:
            li_test.append([df['tran_num'][i], df['tran_sum'][i], df['tran_mean'][i], df['interval_mean'][i], df['interval_var'][i], df['tran_freq'][i], df['tran_num_dir'][i]])

    a_train = np.array(li_train)
    a_test = np.array(li_test)
    a_all = np.array(li_all)
    np.save('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\' + str(i) + '\\train' + '.npy', a_train)
    np.save('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\' + str(i) + '\\test' + '.npy', a_test)
    np.save('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\GAT_preprocess\\' + str(i) + '\\all' + '.npy', a_all)
    # 读取：np.load


# 遍历12个节点对，记录邻接关系
setlist = [] # 储存12个节点对的名字，每个节点对用集合来储存，两个节点的排序没影响，其实12个节点对都是有邻接关系的，因为都是同一个源节点爬取的一阶节点
for j in range(len(name_node_pairs)):
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link features_5_7days\\'+name_node_pairs[j]+'_temp_link_ft.csv')
    df = pd.read_csv(file)
    tempset = set()
    str = name_node_pairs[j]
    index_ = str.find('_')
    a = str[0:index_]
    b = str[index_+1:]
    tempset.add(a)
    tempset.add(b)
    setlist.append(tempset)

graph = defaultdict(list)
for i in range(len(name_node_pairs)):
    for j in range(len(name_node_pairs)):
        if i!=j:
            if setlist[i]&setlist[j]:
                graph[i].append(j)
        else:
            continue
