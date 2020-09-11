import pandas as pd
import numpy as np
# 统计出现大额异动的范围和是否出现大额移动

name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\name_node_pairs_2_quchong_with12_without_notran.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

num = 0
for i in range(len(name_node_pairs)):
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\temporal link features_5_7days_739\\' +
                name_node_pairs[i] + '_temp_link_ft.csv')
    df = pd.read_csv(file)

    tran_sum = df['tran_sum'].values
    tran_sum_3 = tran_sum[:3] # 前三个时间段的交易额

    mean3 = tran_sum_3.mean() # 前三个时间段的交易额的平均
    std3 = tran_sum_3.std() # 前三个时间段的交易额的标准差

    # 大额异动的范围
    min_ = mean3-std3*3
    max_ = mean3+std3*3

    big_change1 = False
    big_change2 = False

    if tran_sum[-2]>max_ or tran_sum[-2]<min_:
        big_change1 = True
    if tran_sum[-1]>max_ or tran_sum[-1]<min_:
        big_change2 = True

    data = [[min_, max_, big_change1, big_change2]]
    df2 = pd.DataFrame(data, columns=['min', 'max', 'big_change1', 'big_change2'])
    df2.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\大额异动\\' +
                name_node_pairs[i] + '.csv')

big_changes = 0
for i in range(len(name_node_pairs)):
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\大额异动\\' +
                name_node_pairs[i] + '.csv')
    df = pd.read_csv(file)
    if df['big_change2'][0] == True or df['big_change1'][0] == True:
        pos = name_node_pairs[i].find('_')
        y = name_node_pairs[i][pos+1:]
        print(y)
        big_changes += 1

print(big_changes)