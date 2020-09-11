# 统计节点对（包括邻接点）的基本交易信息
import pandas as pd
import csv

# # 查看有多少tran_num大于11000的，因为大部分都在10000左右，所以主要的交易数量还是在交易所节点0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be那里
# # 90个大于11000
#
# # 多少大于15000
# # 19
#
# # 多少大于20000
# # 13
# file = open('samples_info.csv')
# df = pd.read_csv(file)
# num = 0
# for i in range(len(df)):
#     if df['transaction_num'][i]>=20000:
#         num+=1
#         print(num)

name = open('name_node_pairs.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

file = open('samples_info.csv','w',newline='')
csvwriter = csv.writer(file)
csvwriter.writerow(['','node_pair','transaction_num','min_timestamp','max_timestamp','time_interval','transaction_frequency'])
num = 0

for i in range(len(name_node_pairs)):
    print(i)
    node_pair = name_node_pairs[i]
    file = open('./0_1/'+node_pair+'.csv')
    df_node_pair = pd.read_csv(file)

    transaction_num = len(df_node_pair)
    min_timestamp = min(df_node_pair['TimeStamp'])
    # print('min_timestamp:'+str(min_timestamp))
    max_timestamp = max(df_node_pair['TimeStamp'])
    # print('max_timestamp:'+str(max_timestamp))
    time_interval = max_timestamp - min_timestamp
    # print('time_interval:'+str(time_interval))
    transaction_frequency = transaction_num/time_interval*86400 # 交易次数除以时间间隔（天数）

    csvwriter.writerow([num, node_pair, transaction_num, min_timestamp, max_timestamp, time_interval, transaction_frequency])
    num = num + 1