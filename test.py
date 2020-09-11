import math
import pandas as pd
import csv

# 查看有多少tran_num大于11000的，因为大部分都在10000左右，所以主要的交易数量还是在交易所节点0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be那里
# 90个大于11000

# 多少大于15000
# 19

# 多少大于20000
# 13
file = open('samples_info.csv')
df = pd.read_csv(file)

file2 = open('name_node_pairs_tran_num_21000.csv','w',newline='')
csvwriter = csv.writer(file2)
csvwriter.writerow(['name_node_pairs'])
num = 0
for i in range(len(df)):
    if df['transaction_num'][i]>=21000:
        num += 1
        print(num)
        csvwriter.writerow([df['node_pair'][i]])





