import pandas as pd
import csv
import numpy as np
import math

name = open('name_node_pairs_selected.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

# 预处理：将交易次数大于10的节点对，按15天为时间间隔从第一个交易值大于零的t开始连续取10个间隔
for i in range(len(name_node_pairs)):
    file = open('./temporal link features_5_node_pairs/' + name_node_pairs[i] + '_temp_link_ft.csv')
    df = pd.read_csv(file)
    df = df.iloc[215:225]

    # 将抽取出10个时间段的文件储存在同一个文件夹下
    df.to_csv('./temporal link features_5_node_pairs/' + name_node_pairs[i] + '.csv', index=False)

# 创建带有时间序号的空的temporal link prediction 文件
for i in range(len(name_node_pairs)):

    file = open('./temporal link prediction_5_node_pairs/'+name_node_pairs[i]+'.csv','w',newline='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['t', 'tran_sum_real', 'tran_sum_la', 'tran_sum_ha', 'difference_la', 'difference_ha'])
    for j in range(2):
        csvwriter.writerow([j+8, 0., 0., 0., 0., 0.])
    file.close()

mse_la = 0.
mse_ha = 0.

for i in range(len(name_node_pairs)):

    print(i)

    # 打开每个节点对的信息后面进行统计
    file = open('./temporal link features_5_node_pairs/' + name_node_pairs[i] + '.csv')
    df_node_pair = pd.read_csv(file)
    file.close()

    # 创建预测文件
    file2 = open('./temporal link prediction_5_node_pairs/' + name_node_pairs[i] + '.csv')
    df_prediction = pd.read_csv(file2)
    file2.close()

    last_value = 0.
    historical_sum = 0.

    for j in range(8):
        tran_sum = df_node_pair['tran_sum'][j]
        last_value = tran_sum
        historical_sum = historical_sum + tran_sum

    for j in range(8, 10):
        tran_sum = df_node_pair['tran_sum'][j]
        df_prediction['tran_sum_real'][j-8] = tran_sum
        df_prediction['tran_sum_ha'][j-8] = historical_sum/j
        historical_sum = historical_sum + df_prediction['tran_sum_ha'][j-8]
        df_prediction['tran_sum_la'][j-8] = last_value
        df_prediction['difference_ha'][j-8] = df_prediction['tran_sum_ha'][j-8] - df_prediction['tran_sum_real'][j-8]
        df_prediction['difference_la'][j-8] = df_prediction['tran_sum_la'][j-8] - df_prediction['tran_sum_real'][j-8]

        # 计算mse，先在循环里累加difference的平方，后面再求均值和开方
        mse_ha = mse_ha + math.pow(df_prediction['difference_ha'][j-8], 2)
        mse_la = mse_la + math.pow(df_prediction['difference_la'][j-8], 2)

    df_prediction.to_csv('./temporal link prediction_5_node_pairs/' + name_node_pairs[i] + '.csv', index=False)

mse_ha = math.sqrt(mse_ha/(5*2))
mse_la = math.sqrt(mse_la/(5*2))
print('mse_ha:'+str(mse_ha))
print('mse_la:'+str(mse_la))
