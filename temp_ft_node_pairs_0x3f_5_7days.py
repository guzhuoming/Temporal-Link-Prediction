# 5个挑选的节点对
import pandas as pd
import csv
import numpy as np
import time
import datetime

name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\name_node_pairs_selected_0x3f_5_7days.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

# 创建带有时间序号的空的temporal link feature 文件
for i in range(len(name_node_pairs)):

    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link features_5_7days\\'+name_node_pairs[i]+'_temp_link_ft.csv','w',newline='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['tran_num', 'tran_sum', 'tran_mean', 'tran_var', 'interval_mean', 'interval_var', 'tran_freq', 'tran_num_dir'])#directional
    for j in range(5):
        csvwriter.writerow([0, 0., 0., 0., 0., 0., 0., 0])
    file.close()

# 进行统计
for i in range(len(name_node_pairs)):
    print('--------------------------------i = '+str(i)+'--------------------------------')
    # 打开每个节点对
    node_pair = name_node_pairs[i]
    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\node_pairs_selected_5_7days\\'+node_pair+'.csv')
    df_node_pair = pd.read_csv(file, index_col=0)
    df_node_pair.index = range(len(df_node_pair))

    timestamp = [[]for i in range(5)]  # 储存交易的时间戳
    tran = [[]for i in range(5)]  # 储存交易额
    for j in range(len(df_node_pair)):

        npos = node_pair.index('_') # “_” 的位置
        x = node_pair[0:npos]
        y = node_pair[npos+1:]
        file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link features_5_7days\\' + name_node_pairs[i] + '_temp_link_ft.csv')
        df = pd.read_csv(file)
        file.close()
        if (df_node_pair['From'][j] == x and df_node_pair['To'][j] == y) or \
                (df_node_pair['From'][j] == y and df_node_pair['To'][j] == x):

            print(name_node_pairs[i])
            t = (df_node_pair['TimeStamp'][j]-1455206400)//(86400*7)
            print('第几条:'+str(j))
            print('t:'+str(t))
            temp_t = t-95
            df['tran_num'][temp_t] = df['tran_num'][temp_t] + 1
            print('tran_num:' + str(df['tran_num'][temp_t]))
            print('tran_sum:'+str(df['tran_sum'][temp_t]))
            if df_node_pair['From'][j] == x and df_node_pair['To'][j] == y:
                df['tran_num_dir'][temp_t] = df['tran_num_dir'][temp_t] + 1
            df['tran_sum'][temp_t] = df['tran_sum'][temp_t] + df_node_pair['Value'][j]
            print('df_node_pair[value][j]:'+str(df_node_pair['Value'][j]))
            print('tran_sum:' + str(df['tran_sum'][temp_t]))
            tran[temp_t].append(df_node_pair['Value'][j])
            timestamp[temp_t].append(df_node_pair['Value'][j])
        df.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link features_5_7days\\' + name_node_pairs[i] + '_temp_link_ft.csv', index=False)

    for t in range(5):
        if len(tran[t])>0:
            df['tran_mean'][t] = np.mean(tran[t])
            df['tran_var'][t] = np.var(tran[t])
            df.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link features_5_7days\\' + name_node_pairs[i] + '_temp_link_ft.csv', index=False)
        if len(timestamp[t]) > 1:
            temp = [] # 储存交易时间间隔
            for n in range(len(timestamp[t])-1):
                temp = temp + [timestamp[t][n+1]-timestamp[t][n]]
                df['interval_mean'][t] = np.mean(temp)
                df['interval_var'][t] = np.var(temp)
            df.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link features_5_7days\\' + name_node_pairs[i] + '_temp_link_ft.csv', index=False)
    #-----------------------交易频率-------------------
    #暂定是交易次数每周
    df['tran_freq'] = df['tran_num']
    #--------------------------------------------------
    df.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\temporal link features_5_7days\\'+name_node_pairs[i]+'_temp_link_ft.csv', index=False)
    print('成功！')