import pandas as pd
import csv
import numpy as np
import math

name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\name_node_pairs_selected_0x3f_5_7days.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

# 创建带有时间序号的空的temporal link prediction 文件
for i in range(len(name_node_pairs)):

    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link prediction_12nodepairs_LA_HA\\'+name_node_pairs[i]+'.csv','w',newline='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['t', 'tran_sum_real', 'tran_sum_la', 'tran_sum_ha', 'difference_la', 'difference_ha'])
    for j in range(2):
        csvwriter.writerow([j+3, 0., 0., 0., 0., 0.])
    file.close()

mse_la = 0.
mse_ha = 0.

for i in range(len(name_node_pairs)):

    # 打开每个节点对的信息后面进行统计
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link features_5_7days\\' + name_node_pairs[i] + '_temp_link_ft.csv')
    df_node_pair = pd.read_csv(file)
    file.close()

    # 创建预测文件
    file2 = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link prediction_12nodepairs_LA_HA\\' + name_node_pairs[i] + '.csv')
    df_prediction = pd.read_csv(file2)
    file2.close()

    last_value = 0.
    historical_sum = 0.
    tran_sum_acc = 0 #累计的交易总额

    for j in range(3):
        tran_sum_acc = tran_sum_acc +  df_node_pair['tran_sum'][j]
        # last_value = tran_sum_acc
        last_value = df_node_pair['tran_sum'][j]

    historical_sum = tran_sum_acc
    for j in range(3, 5):
        tran_sum = df_node_pair['tran_sum'][j]
        if tran_sum>0:
            print(name_node_pairs[i])

        df_prediction['tran_sum_real'][j-3] = tran_sum
        df_prediction['tran_sum_ha'][j-3] = historical_sum/j
        tran_sum_acc = tran_sum_acc + tran_sum
        historical_sum = historical_sum + df_prediction['tran_sum_ha'][j-3]
        df_prediction['tran_sum_la'][j-3] = last_value
        df_prediction['difference_ha'][j-3] = df_prediction['tran_sum_ha'][j-3] - df_prediction['tran_sum_real'][j-3]
        df_prediction['difference_la'][j-3] = df_prediction['tran_sum_la'][j-3] - df_prediction['tran_sum_real'][j-3]

        # 计算mse，先在循环里累加difference的平方，后面再求均值和开方
        mse_ha = mse_ha + math.pow(df_prediction['difference_ha'][j-3], 2)
        mse_la = mse_la + math.pow(df_prediction['difference_la'][j-3], 2)
        print('mse_ha')
        print(mse_ha)
        print('mse_la')
        print(mse_la)

    df_prediction.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link prediction_12nodepairs_LA_HA\\' + name_node_pairs[i] + '.csv', index=False)

mse_ha = math.sqrt(mse_ha/(len(name_node_pairs)*2))
mse_la = math.sqrt(mse_la/(len(name_node_pairs)*2))
print('mse_ha:'+str(mse_ha))
print('mse_la:'+str(mse_la))