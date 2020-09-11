# 以1天为时间间隔，对挑选的5个节点对进行可视化预处理

import pandas as pd
import numpy as np
import csv
import math

name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\name_node_pairs_2_quchong_with12_without_notran.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

for t in range(5):
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\Gephi\\'+str(t)+'.csv','w',newline='')
    csvwriter = csv.writer(file)
    #需要将from 和 to改成source和target
    csvwriter.writerow(['Source', 'Target', 'Value'])

for i in range(len(name_node_pairs)):
    node_pair = name_node_pairs[i]
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\0_1_quchong_and_12\\'+node_pair+'.csv')
    df_node_pair = pd.read_csv(file)
    print(i)
    for j in range(len(df_node_pair)):
        t = (df_node_pair['TimeStamp'][j]-1455206400)//(86400*7)
        if t>=95 and t<100:
            Source = df_node_pair['From'][j]
            Target = df_node_pair['To'][j]
            Value = df_node_pair['Value'][j]
            file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\Gephi\\' + str(t) + '.csv', 'a+', newline='')
            csvwriter = csv.writer(file)
            csvwriter.writerow([Source, Target, Value])

for t in range(95, 100):
    print('t='+str(t))
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\Gephi\\'+str(t)+'.csv')
    df = pd.read_csv(file)
    df['duplicate'] = len(df)*[0]#判断重复的是否已经累加
    if len(df)>1:
        for j in range(len(df)):
            if df['duplicate'][j]==1:
                continue
            for k in range(j+1,len(df)):
                if df['Source'][j]==df['Source'][k] and df['Target'][j]==df['Target'][k] and df['duplicate'][k]==0:
                    df['Value'][j] = df['Value'][j] + df['Value'][k]
                    df['duplicate'][k] = 1

    df = df[df['duplicate']==0]
    df.drop(['duplicate'], axis=1, inplace=True)
    df.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\Gephi\\'+str(t)+'_合并后.csv',index=False)