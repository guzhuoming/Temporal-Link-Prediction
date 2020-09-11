# 此文件先从我们之前爬取的label文件中挑取txn count最大的节点，再判断其是否再ht的0阶节点数据集中
import pandas as pd
import csv
import numpy as np

#---------------------- 取得所有label并且读取改label的csv
file1 = open('E:\\以太坊label_csv\\html_accounts.csv')
df1 = pd.read_csv(file1)
# 储存最大的TxnCount对应的Address
maxCount = 0
maxAddress = ''
maxLabel = ''
for i in range(len(df1)):
    print(i)
    print(df1['html_accounts'][i])
    label = df1['html_accounts'][i][36:]
    file2 = open('E:\\etherscan_label\\accounts\\'+label+'.csv')
    df2 = pd.read_csv(file2)
    for j in range(len(df2)):
        count = 0
        if type(df2['Txn Count'][j]) == str:
            count = int(df2['Txn Count'][j].replace(',', ''))
        else:
            count = df2['Txn Count'][j]
        if count > maxCount:
            maxCount = count
            maxAddress = df2['Address'][j]
            maxLabel = df1['html_accounts'][i]
print(maxAddress)
print(maxLabel)

# 我们挑取的节点是 0xea674fdde714fd979de3edf0f56aa9716b898ec8
# 但是这个节点是mining节点
# 找交易所的节点可能会好一点，最大Txncount的节点是 0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be