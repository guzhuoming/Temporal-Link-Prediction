# 看一下1740个节点对在95-99的五个时间段有多少节点是没有交易的，删去
import pandas as pd
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import random

# 以100天为时间间隔，储存，每个小括号里是1811个节点对
s = [[]for i in range(1408)]

name = open('./data/name_node_pairs_2_quchong_with12.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

# li = [1,5,7,10,18,20,22,23,25,28,49,58]
# node_pairs_selected = pd.DataFrame(columns=['name_node_pairs'])
# for i in li:
#     node_pairs_selected = node_pairs_selected.append({'name_node_pairs':name_node_pairs[i]}, ignore_index=True)
# node_pairs_selected.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\name_node_pairs_selected_0x3f_5_7days.csv', index=False)
# name_node_pairs_selected = node_pairs_selected['name_node_pairs']

#将挑选出的交易记录保存在一个文件夹里
for i in range(len(name_node_pairs)):
    print(i)
    print(name_node_pairs[i])
    file = open('./data/0_1_quchong_and_12/'+name_node_pairs[i]+'.csv')
    df = pd.read_csv(file)
    df.to_csv('./data/node_pairs_selected_7days/'+name_node_pairs[i]+'.csv', index=False)
# 只保留这些节点在95-99时的值
# 95*7*86400+1455206400 = 1512662400
# 100*7*86400+1455206400 = 1515686400

num = 0 # 用来统计5个时间段都没有交易的节点对有多少
no_tran_list = []
for i in range(len(name_node_pairs)):
    print(i)
    file = open('./data/0_1_quchong_and_12/' + name_node_pairs[i] + '.csv')
    df = pd.read_csv(file)
    df = df[(df['TimeStamp']>=1512662400)&(df['TimeStamp']<1515686400)]
    if(len(df) == 0):
        num+=1
        print(num)
        no_tran_list.append(i)
    else:
        df.to_csv('./data/node_pairs_selected_5_7days/' + name_node_pairs[i] + '.csv', index=False)

print('------------num------------')
print(num)
print(no_tran_list)
# 创建带有时间序号的空的temporal link prediction 文件
# for i in range(len(name_node_pairs)):
#     print(i)
#     file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\0_1\\'+name_node_pairs[i]+'.csv')
#     df_node_pair = pd.read_csv(file)
#     print(name_node_pairs[i])
#     node_pair = name_node_pairs[i]
#     for j in range(len(df_node_pair)):
#         npos = node_pair.index('_')  # “_” 的位置
#         x = node_pair[0:npos]
#         y = node_pair[npos + 1:]
#         if (df_node_pair['From'][j] == x and df_node_pair['To'][j] == y) or \
#                 (df_node_pair['From'][j] == y and df_node_pair['To'][j] == x):
#             t = (df_node_pair['TimeStamp'][j] - 1455206400) // (86400*7)
#             s[t].append(i+random.uniform(-0.15, 0.15))
#
# for t in range(1408):
#     print(t)
#     plt.scatter(len(s[t])*[[t]], s[t], c='b', marker='x')
#
# plt.xlabel('time')
# plt.ylabel('node_pairs')
# plt.grid(axis='y')
# plt.axis([-1, 1408, -1, 90])
# plt.show()

