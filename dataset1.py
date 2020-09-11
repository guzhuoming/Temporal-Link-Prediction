# 合并节点对的交易记录
import pandas as pd
import csv
import os
# 之前选取的源节点为0xea674fdde714fd979de3edf0f56aa9716b898ec8
# 先做10个节点对（0xea674fdde714fd979de3edf0f56aa9716b898ec8和10个一阶节点构成的10个节点对）

file = open('E:\\ht\\data-0-hop-with-label-816\\0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be.csv')
df = pd.read_csv(file)

# 对源节点进行清洗
# 1.from或者to为null时，contractAddress不为null，好像交易额都是0，我们不考虑这种，将其删去
# 2.将isError=1的删除

df.drop(['Unnamed: 0','BlockHeight', 'ContractAddress'], axis = 1, inplace=True)
df = df[df['isError']==0]
df.index = range(len(df))

temp_ = []
for i in range(len(df)):
    if type(df['From'][i]) == float or type(df['To'][i]) == float:
        temp_.append(i)
df.drop(temp_, inplace=True)
df.index = range(len(df))

num = 0
i = 0
set_1hop = []#list 储存10个一阶节点（黄涛对这10个节点爬取了所有二阶节点）

while i<len(df):
    if df['To'][i] == '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be' and df['From'][i] not in set_1hop:
        set_1hop.append(df['From'][i])
        num += 1
    if df['From'][i] == '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be' and df['To'][i] not in set_1hop:
        set_1hop.append(df['To'][i])
        num += 1
    i += 1
    print('num='+str(num))

df_set_1_hop_name = pd.DataFrame(columns=['name'])

num = 0 #这个num用于后面保存节点对的名字
file_ = open('F:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\name_node_pairs_2.csv','w',newline='') #2----包括二阶节点
csvwriter = csv.writer(file_)
csvwriter.writerow(['','name_node_pairs'])

for i in range(len(set_1hop)):
    df_set_1_hop_name = df_set_1_hop_name.append([{'name':set_1hop[i]}], ignore_index=True)

df_set_1_hop_name.to_csv('F:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\1_hop_name.csv', index=False)

for i in range(len(set_1hop)):
    print(i)
    print(set_1hop[i])

    # 判断文件在哪
    if os.path.exists('E:\\ht\\data-only-1-hop-without0-639668\\'+set_1hop[i]+'.csv'):
        print(1)
        file_1hop = open('E:\\ht\\data-only-1-hop-without0-639668\\'+set_1hop[i]+'.csv')
        df_1_hop = pd.read_csv(file_1hop)

        # 数据清洗
        # 1.from或者to为null时，contractAddress不为null，好像交易额都是0，我们不考虑这种，将其删去
        # 2.将isError=1的删除
        df_1_hop.drop(['Unnamed: 0', 'BlockHeight', 'ContractAddress', 'Input'], axis=1, inplace=True)
        df_1_hop = df_1_hop[df_1_hop['isError']==0]
        df_1_hop.drop(['isError'], axis=1, inplace=True)
        df_1_hop.index=range(len(df_1_hop))#否则因为有些index被删掉，后面按index取值会出现问题
    elif os.path.exists('E:\\ht\\data-2-hop\\'+set_1hop[i]+'.csv'):
        print(2)
        file_1hop = open('E:\\ht\\data-2-hop\\' + set_1hop[i] + '.csv')
        df_1_hop = pd.read_csv(file_1hop)

        # 数据清洗
        # 1.from或者to为null时，contractAddress不为null，好像交易额都是0，我们不考虑这种，将其删去
        # 2.将isError=1的删除
        df_1_hop.drop(['Unnamed: 0', 'BlockHeight', 'ContractAddress', 'Input'], axis=1, inplace=True)
        df_1_hop = df_1_hop[df_1_hop['isError'] == 0]
        df_1_hop.drop(['isError'], axis=1, inplace=True)
    elif os.path.exists('E:\\ht\\data-0-hop-with-label-816\\'+set_1hop[i]+'.csv'):
        print(0)
        file_1hop = open('E:\\ht\\data-0-hop-with-label-816\\' + set_1hop[i] + '.csv')
        df_1_hop = pd.read_csv(file_1hop)

        # 数据清洗
        # 1.from或者to为null时，contractAddress不为null，好像交易额都是0，我们不考虑这种，将其删去
        # 2.将isError=1的删除
        df_1_hop.drop(['Unnamed: 0', 'BlockHeight', 'ContractAddress'], axis=1, inplace=True)
        df_1_hop = df_1_hop[df_1_hop['isError'] == 0]
        df_1_hop.drop(['isError'], axis=1, inplace=True)
    else:
        continue

    temp = [] #后面的循环里储存from和to有空的index，循环后一起删除
    print(len(df_1_hop))
    df_1_hop.index = range(len(df_1_hop))
    for j in range(len(df_1_hop)):
        print(j)
        if type(df_1_hop['From'][j]) == float or type(df_1_hop['To'][j]) == float:
            temp.append(j)
    df_1_hop.drop(temp, inplace=True)
    df_1_hop.index = range(len(df_1_hop))
    df_0_1 = pd.concat([df, df_1_hop], axis=0, join='outer', ignore_index=True)
    df_0_1.drop_duplicates(inplace=True)
    df_0_1.sort_values(by='TimeStamp', inplace=True)
    df_0_1.index = range(len(df_0_1))#对序号重新排序

    df_0_1.to_csv('F:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\0_1\\' + '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be' + '_' + set_1hop[i] + '.csv')
    csvwriter.writerow([num, '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be' + '_' + set_1hop[i]])
    num+=1
    # --------------------------------------------------------------------
    # 1-2hop
    # 记录所有的一阶节点i的所有二阶节点
    set_2_hop = []
    for j in range(len(df_1_hop)):
        print('i'+str(i)+'j'+str(j))
        if df_1_hop['From'][j] != '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be' and df_1_hop['From'][j] != set_1hop[i]:
            if df_1_hop['From'][j] not in set_2_hop:
                set_2_hop.append(df_1_hop['From'][j])
        if df_1_hop['To'][j] != '0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be' and df_1_hop['To'][j] != set_1hop[i]:
            if df_1_hop['To'][j] not in set_2_hop:
                set_2_hop.append(df_1_hop['To'][j])
    for j in range(len(set_2_hop)):
        print('二阶'+str(j))
        print(set_2_hop[j])
        # 二阶节点不一定在二阶文件夹里，可能在0，1，2阶文件夹里，只有0阶文件夹里的文件没有input，所以drop时要分情况
        if os.path.exists('E:\\ht\\data-2-hop\\' + set_2_hop[j] + '.csv'):
            print(2)
            file_2hop = open('E:\\ht\\data-2-hop\\' + set_2_hop[j] + '.csv')
            df_2_hop = pd.read_csv(file_2hop)
            # 数据清洗
            df_2_hop.drop(['Unnamed: 0', 'BlockHeight', 'ContractAddress', 'Input'], axis=1, inplace=True)

        elif os.path.exists('E:\\ht\\data-only-1-hop-without0-639668\\' + set_2_hop[j] + '.csv'):
            print(1)
            file_2hop = open('E:\\ht\\data-only-1-hop-without0-639668\\' + set_2_hop[j] + '.csv')
            df_2_hop = pd.read_csv(file_2hop)
            # 数据清洗
            df_2_hop.drop(['Unnamed: 0', 'BlockHeight', 'ContractAddress', 'Input'], axis=1, inplace=True)

        elif os.path.exists('E:\\ht\\data-0-hop-with-label-816\\' + set_2_hop[j] + '.csv'):
            print(0)
            file_2hop = open('E:\\ht\\data-0-hop-with-label-816\\' + set_2_hop[j] + '.csv')
            df_2_hop = pd.read_csv(file_2hop)
            # 数据清洗
            df_2_hop.drop(['Unnamed: 0', 'BlockHeight', 'ContractAddress'], axis=1, inplace=True)

        else:#还有找不到节点的情况
            print('notfound')
            continue

        df_2_hop = df_2_hop[df_2_hop['isError']==0]
        df_2_hop.drop(['isError'], axis=1, inplace=True)
        df_2_hop.index = range(len(df_2_hop))

        temp = []

        for k in range(len(df_2_hop)):
            if type(df_2_hop['From'][k]) == float or type(df_2_hop['To'][k]) == float:
                temp.append(k)
        df_2_hop.drop(temp, inplace=True)
        df_2_hop.index = range(len(df_2_hop))
        df_1_2 = pd.concat([df_1_hop, df_2_hop], axis=0, join='outer', ignore_index=True)
        df_1_2.drop_duplicates(inplace=True)
        df_1_2.sort_values(by='TimeStamp', inplace=True)
        df_1_2.index = range(len(df_1_2))

        df_1_2.to_csv('F:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_90\\1_2\\'+set_1hop[i]+'_'+set_2_hop[j]+'.csv')
        csvwriter.writerow([num, set_1hop[i]+'_'+set_2_hop[j]])
        num+=1

print('num: '+str(num))