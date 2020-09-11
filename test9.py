# 画交易金额分布图
import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import time
import datetime
from collections import defaultdict
import networkx as nx


# ====================打开文件============================

name = open('./data/name_node_pairs_2_quchong_with12_without_notran.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

# ==================统计信息，话直方图=======================

total_list = []
print('统计进度：')
for i in range(len(name_node_pairs)):
    print(str(i/len(name_node_pairs)*100)+'%')
    file = open('./data/temporal link features_5_7days_739/'+name_node_pairs[i]+'_temp_link_ft.csv')
    df = pd.read_csv(file)
    tran_sum = df['tran_sum'].values
    tran_sum = tran_sum.tolist()
    total_list = total_list+tran_sum

total_list = np.array(total_list)
np.save('./data/total_list.npy', total_list) # 保存后就不用每次都运行了
total_list = total_list.tolist()

total_list_ = [] # total_list里面存在一些交易金额为零的，去除他
for i in range(len(total_list)):
    if total_list[i] > 0:
        total_list_.append(total_list[i])
        # print(name_node_pairs[i//5])
print(len(total_list_))
# 1256

# print(max(total_list_))
# 20000

num_groups = 1000
plt.hist(total_list_, num_groups, rwidth=0.8, density=False, cumulative=False)
plt.xlabel('Value')
plt.ylabel('Probability')
plt.title('Distribution of Transaction Amount')
plt.show()



# ==================================================================
