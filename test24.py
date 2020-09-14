# 检查三个最大交易额的节点对
import pandas as pd
import csv
import time
import datetime

strs = ['0x30b10221c90134db4bb8388748428bc0778594d8_0x8b1698f7344d8446821940dfa180dd0f6eb03ca9',
        '0x4be2166a8dbe25e79a9b30b4ce26858e10d21e92_0xafba1b04ec9dcbdeaa8601d8ae8f44edc77da224',
        '0x43b687b0a11718730c562ee32a103d93bfe458bc_0xd37cff84883f8b92e778b7dac0ca354e28741e09']

for i in range(len(strs)):
    file = open('./data/0_1/' + strs[i] + '.csv')
    df = pd.read_csv(file)
    pos = strs[i].find('_')
    x = strs[i][0:pos]
    print(x)
    y = strs[i][pos+1:]
    print(y)

    df = df[(df['TimeStamp'] >= 1512662400) & (df['TimeStamp'] < 1515686400)] # 2017-12-08 00:00:00  2017-12-15  2017-12-22   2017-12-29   2018-01-05  2018-01-12
    df1 = df[(df['From'] == x) & (df['To'] == y)]
    df2 = df[(df['To'] == y) & (df['From'] == x)]
    df = pd.concat([df1,df2])
    df = df.drop(['Unnamed: 0', 'isError'], axis=1)
    df = df.drop_duplicates()

    time_ = []
    df.index = range(len(df))

    for j in range(len(df['TimeStamp'])):
        timeStamp = df['TimeStamp'][j]
        timeArray = time.localtime(timeStamp)
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        time_.append(otherStyleTime)

    df['time'] = time_

    df.to_csv('./data/3_biggest/' + strs[i] + '.csv', index=False)