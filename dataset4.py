# 5个挑选的节点对
import pandas as pd
import csv
import numpy as np
import time
import datetime

name = open('name_node_pairs_tran_num_21000.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

# 创建带有时间序号的空的temporal link feature 文件
for i in range(len(name_node_pairs)):

    file = open('./temporal link features_83_node_pairs/'+name_node_pairs[i]+'_temp_link_ft.csv','w',newline='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['tran_num', 'tran_sum', 'tran_mean', 'tran_var', 'interval_mean', 'interval_var', 'tran_freq', 'tran_num_dir'])#directional
    for j in range(1604):
        csvwriter.writerow([0, 0., 0., 0., 0., 0., 0., 0])
    file.close()

# 进行统计
for i in range(len(name_node_pairs)):

    print(i)
    # 打开每个节点对
    node_pair = name_node_pairs[i]
    file = open('./0_1/'+node_pair+'.csv')
    df_node_pair = pd.read_csv(file, index_col=0)

    timestamp = [[]for i in range(1604)]  # 储存交易的时间戳
    tran = [[]for i in range(1604)]  # 储存交易额
    for j in range(len(df_node_pair)):

        npos = node_pair.index('_') # “_” 的位置
        x = node_pair[0:npos]
        y = node_pair[npos+1:]
        file = open('./temporal link features_83_node_pairs/' + name_node_pairs[i] + '_temp_link_ft.csv')
        df = pd.read_csv(file)
        file.close()

        if (df_node_pair['From'][j] == x and df_node_pair['To'][j] == y) or \
                (df_node_pair['From'][j] == y and df_node_pair['To'][j] == x):

            print(name_node_pairs[i])
            t = (df_node_pair['TimeStamp'][j]-1438185600)//(86400)
            print('第几条:'+str(j))
            print('t:'+str(t))
            df['tran_num'][t] = df['tran_num'][t] + 1
            print('tran_num:' + str(df['tran_num'][t]))
            print('tran_sum:'+str(df['tran_sum'][t]))
            if df_node_pair['From'][j] == x and df_node_pair['To'][j] == y:
                df['tran_num_dir'][t] = df['tran_num_dir'][t] + 1
            df['tran_sum'][t] = df['tran_sum'][t] + df_node_pair['Value'][j]
            print('df_node_pair[value][j]:'+str(df_node_pair['Value'][j]))
            print('tran_sum:' + str(df['tran_sum'][t]))
            tran[t].append(df_node_pair['Value'][j])
            timestamp[t].append(df_node_pair['Value'][j])
        df.to_csv('./temporal link features_83_node_pairs/' + name_node_pairs[i] + '_temp_link_ft.csv', index=False)

    for t in range(1604):
        if len(tran[t])>0:
            df['tran_mean'][t] = np.mean(tran[t])
            df['tran_var'][t] = np.var(tran[t])
            df.to_csv('./temporal link features_83_node_pairs/' + name_node_pairs[i] + '_temp_link_ft.csv', index=False)
        if len(timestamp[t]) > 1:
            temp = [] # 储存交易时间间隔
            for n in range(len(timestamp[t])-1):
                temp = temp + [timestamp[t][n+1]-timestamp[t][n]]
                df['interval_mean'][t] = np.mean(temp)
                df['interval_var'][t] = np.var(temp)
            df.to_csv('./temporal link features_83_node_pairs/' + name_node_pairs[i] + '_temp_link_ft.csv', index=False)
    #-----------------------交易频率-------------------
    #暂定是交易次数每天
    df['tran_freq'] = df['tran_num']
    #--------------------------------------------------
    df.to_csv('./temporal link features_83_node_pairs/'+name_node_pairs[i]+'_temp_link_ft.csv', index=False)
    print('成功！')