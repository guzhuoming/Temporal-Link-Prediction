# 这个还没有运行，没有确定时间间隔，因为是从源节点出发的整个01，12阶节点对

import pandas as pd
import csv
import time
import datetime
name = open('name_node_pairs_tran_num_21000.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

# 查询最小时间和最大时间，以一天为时间间隔划分
file = open('samples_info.csv')
df_samples_info = pd.read_csv(file)
samples_info = df_samples_info['min_timestamp']
min_time = min(samples_info)
samples_info = df_samples_info['max_timestamp']
max_time = max(samples_info)
print('min_time:'+str(min_time))
print('max_time:'+str(max_time))
# 显示日期
dateArray = datetime.datetime.fromtimestamp(min_time)
otherStyleTime = dateArray.strftime("%Y--%m--%d %H:%M:%S")
print(otherStyleTime)
# 2015--07--30 23:26:13 min_time:1438269973
dateArray = datetime.datetime.fromtimestamp(max_time)
otherStyleTime = dateArray.strftime("%Y--%m--%d %H:%M:%S")
# 2019--12--19 14:05:56 max_time:1576735556
print(otherStyleTime)
# 2015--07--30 00:00:00     1438185600
# 2019--12--19 00:00:00     1576684800
# 每天的间隔为86400秒
# (1576684800-1438185600)/86400 = 1603.0 共有1604个Gt_edgelist


# for t in range(15):
#     file = open('./Gt_egdelist/G'+str(t)+'_egdelist.csv','w',newline='')
#     csvwriter = csv.writer(file)
#     csvwriter.writerow(['From','To','Value'])
#
# for i in range(len(name_node_pairs)):
#     node_pair = name_node_pairs[i]
#     file = open('./合并节点对的交易记录/'+node_pair+'.csv')
#     df_node_pair = pd.read_csv(file)
#     print(i)
#     for j in range(len(df_node_pair)):
#         t = (df_node_pair['TimeStamp'][j]-1438876800)//(86400*100)
#         From = df_node_pair['From'][j]
#         To = df_node_pair['To'][j]
#         Value = df_node_pair['Value'][j]
#         file = open('./Gt_egdelist/G'+str(t)+'_egdelist.csv','a+',newline='')
#         csvwriter = csv.writer(file)
#         csvwriter.writerow([From, To, Value])
#
# for t in range(15):
#     file = open('./Gt_egdelist/G'+str(t)+'_egdelist.csv',newline='')
#     df = pd.read_csv(file)
#     df.index = range(len(df))
#     df.to_csv('./Gt_egdelist_index/G'+str(t)+'_egdelist.csv')