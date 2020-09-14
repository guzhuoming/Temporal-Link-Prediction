# 去重
import pandas as pd
import csv
import time
import datetime

name = open('./data/name_node_pairs_2_quchong_with12_without_notran.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

for i in range(len(name_node_pairs)):
    print(i)
    print(name_node_pairs[i])
    file = open('./data/node_pairs_selected_5_7days/' + name_node_pairs[i] + '.csv')
    df = pd.read_csv(file)
    try:
        df = df.drop(['isError'], axis=1)
    except:
        print('ex')
    try:
        df = df.drop(['Unnamed: 0'], axis=1)
    except:
        print('ex')
    df = df.drop_duplicates()
    df.index = range(len(df))
    df.to_csv('./data/0_1_quchong_and_12/' + name_node_pairs[i] + '.csv')
