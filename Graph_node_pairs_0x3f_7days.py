# 这个还没有运行，没有确定时间间隔，因为是从源节点出发的整个01，12阶节点对

import pandas as pd
import csv
import time
import datetime
name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\name_node_pairs_selected_0x3f_5_7days.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

for t in range(95,100):
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\Gt_egdelist\\G'+str(t)+'_egdelist.csv','w',newline='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['From','To','Value'])
#
for i in range(len(name_node_pairs)):
    node_pair = name_node_pairs[i]
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\node_pairs_selected_5_7days\\'+node_pair+'.csv')
    df_node_pair = pd.read_csv(file)
    print(i)
    for j in range(len(df_node_pair)):
        print('j: '+str(j))
        t = (df_node_pair['TimeStamp'][j]-1455206400)//(86400*7)
        print('t'+str(t))
        From = df_node_pair['From'][j]
        To = df_node_pair['To'][j]
        Value = df_node_pair['Value'][j]
        file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\Gt_egdelist\\G'+str(t)+'_egdelist.csv','a+',newline='')
        csvwriter = csv.writer(file)
        csvwriter.writerow([From, To, Value])
#
# for t in range(15):
#     file = open('./Gt_egdelist/G'+str(t)+'_egdelist.csv',newline='')
#     df = pd.read_csv(file)
#     df.index = range(len(df))
#     df.to_csv('./Gt_egdelist_index/G'+str(t)+'_egdelist.csv')