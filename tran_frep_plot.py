import pandas as pd
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import random

# 以100天为时间间隔，储存，每个小括号里是1811个节点对
s = [[]for i in range(1408)]

name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\name_node_pairs_2_quchong_with12.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

# 创建带有时间序号的空的temporal link prediction 文件
for i in range(len(name_node_pairs)):
    print(i)
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\0_1_quchong_and_12\\'+name_node_pairs[i]+'.csv')
    df_node_pair = pd.read_csv(file)
    print(name_node_pairs[i])
    node_pair = name_node_pairs[i]
    for j in range(len(df_node_pair)):
        npos = node_pair.index('_')  # “_” 的位置
        x = node_pair[0:npos]
        y = node_pair[npos + 1:]
        if (df_node_pair['From'][j] == x and df_node_pair['To'][j] == y) or \
                (df_node_pair['From'][j] == y and df_node_pair['To'][j] == x):
            t = (df_node_pair['TimeStamp'][j] - 1455206400) // (86400*7)
            s[t].append(i+random.uniform(-0.15, 0.15))

for t in range(1408):
    print(t)
    plt.scatter(len(s[t])*[[t]], s[t], c='b', marker='x')

plt.xlabel('time')
plt.ylabel('node_pairs')
plt.grid(axis='y')
plt.axis([-1, 1408, -1, 2000])
plt.show()

