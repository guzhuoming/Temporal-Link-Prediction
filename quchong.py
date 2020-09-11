# ---------------对12个节点对扩展的数据集做去重，文件已经手动去重，将namenodepairs_2.csv文件里的namenodepair去重
import csv
import os
import pandas as pd
str = '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be'
file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\name_node_pairs_2.csv')
df = pd.read_csv(file)

droplist = []

for i in range(len(df)):
    name = df['name_node_pairs'][i]
    pos = name.find('_')
    x = name[0:pos]
    y = name[pos + 1:]
    if y == str:
        print(i)
        droplist.append(i)

df.drop(droplist, inplace= True)
df.drop(['Unnamed: 0'], axis = 1, inplace= True)
df.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\name_node_pairs_2_quchong.csv',index=False)
print(len(df))