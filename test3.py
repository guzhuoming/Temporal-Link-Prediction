#--------为了取出90个节点对
import math
import pandas as pd
import csv

# 查看有多少tran_num大于11000的，因为大部分都在10000左右，所以主要的交易数量还是在交易所节点0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be那里
# 90个大于11000

# 多少大于15000
# 19

# 多少大于20000
# 13

file2 = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\name_node_pairs_tran_num_11000.csv')
df = pd.read_csv(file2)
name = df['name_node_pairs']

for i in range(len(name)):
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\0_1\\'+name[i]+'.csv')
    df1 = pd.read_csv(file)
    df1.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\0_1_90\\'+name[i]+'.csv', index=False)


