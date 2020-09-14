# 计算各个模型的precision，recall，F1
# 大额异动个数308
import pandas as pd
import numpy as np

name = open('./data/name_node_pairs_2_quchong_with12_without_notran.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

pred_la = 0 # 记录预测为1的个数
pred_ha = 0
pred_arima = 0
pred_lstm = 0
pred_node2vec_lstm = 0
pred_node2vec = 0
pred_gcn = 0
pred_gat = 0
pred_gcn_lstm = 0
pred_gat_lstm = 0
pred_gat_lstm_gat = 0
pred_node2vec_lstm_att = 0
pred_gcn_lstm_att = 0
pred_gat_lstm_att = 0

real = 574   #真实的为1的个数 big1 308  big2 266

acc_la = 0  # 预测为1且正确的个数
acc_ha = 0
acc_arima = 0
acc_lstm = 0
acc_node2vec_lstm = 0
acc_node2vec = 0
acc_gcn = 0
acc_gat = 0
acc_gcn_lstm = 0
acc_gat_lstm = 0
acc_gat_lstm_gat = 0
acc_node2vec_lstm_att = 0
acc_gcn_lstm_att = 0
acc_gat_lstm_att = 0

file_node2vec_lstm = open('./data/temporal link prediction_node2vec_LSTM/prediciton_node2vec_LSTM.csv')
df_node2vec_lstm = pd.read_csv(file_node2vec_lstm)
file_node2vec = open('./data/temporal link prediction_node2vec/prediciton_node2vec.csv')
df_node2vec = pd.read_csv(file_node2vec)
file_gcn = open('./data/temporal link prediction_GCN/prediciton_GCN.csv')
df_gcn = pd.read_csv(file_gcn)
file_gat = open('./data/temporal link prediction_GAT/prediciton_GAT.csv')
df_gat = pd.read_csv(file_gat)
file_gcn_lstm = open('./data/temporal link prediction_GCN_LSTM/prediciton_GCN_LSTM.csv')
df_gcn_lstm = pd.read_csv(file_gcn_lstm)
file_gat_lstm = open('./data/temporal link prediction_GAT_LSTM/prediciton_GAT_LSTM.csv')
df_gat_lstm = pd.read_csv(file_gat_lstm)
file_gat_lstm_gat = open('./data/temporal link prediction_GAT_LSTM_GAT/prediciton_GAT_LSTM_GAT.csv')
df_gat_lstm_gat = pd.read_csv(file_gat_lstm_gat)

file_node2vec_lstm_att = open('./data/temporal link prediction_node2vec_LSTM_att/prediciton_node2vec_LSTM_att.csv')
df_node2vec_lstm_att = pd.read_csv(file_node2vec_lstm_att)
file_gcn_lstm_att = open('./data/temporal link prediction_GCN_LSTM_att/prediciton_GCN_LSTM_att.csv')
df_gcn_lstm_att = pd.read_csv(file_gcn_lstm_att)
file_gat_lstm_att = open('./data/temporal link prediction_GAT_LSTM_att/prediciton_GAT_LSTM_att.csv')
df_gat_lstm_att = pd.read_csv(file_gat_lstm_att)

for i in range(len(name_node_pairs)):

    file = open('./data/大额异动/' +
                name_node_pairs[i] + '.csv')
    df_bigchange = pd.read_csv(file)
    min_ = df_bigchange['min'][0]
    max_ = df_bigchange['max'][0]
    big_change = []
    big_change.append(df_bigchange['big_change1'][0])
    big_change.append(df_bigchange['big_change2'][0])

    file1 = open('./data/temporal link prediction_LA_HA/' +
                name_node_pairs[i] + '.csv')
    df_laha = pd.read_csv(file1)

    file2 = open('./data/temporal link prediction_ARIMA/' +
                name_node_pairs[i] + '.csv')
    df_arima = pd.read_csv(file2)
    file3 = open('./data/temporal link prediction_LSTM_withoutNegative/' +
                name_node_pairs[i] + '.csv')
    df_lstm = pd.read_csv(file3)

    for j in range(2):
        if df_laha['tran_sum_la'][j]>max_ or df_laha['tran_sum_la'][j]<min_ :
            pred_la += 1
            if big_change[j]:
                acc_la += 1
        if df_laha['tran_sum_ha'][j]>max_ or df_laha['tran_sum_ha'][j]<min_ :
            pred_ha += 1
            if big_change[j] :
                acc_ha += 1
        if df_arima['prediction_ARIMA'][j]>max_ or df_arima['prediction_ARIMA'][j]<min_ :
            pred_arima += 1
            if big_change[j] or big_change[j+1]:
                acc_arima += 1
        if df_lstm['prediction_LSTM'][j]>max_ or df_lstm['prediction_LSTM'][j]<min_ :
            pred_lstm += 1
            if big_change[j] :
                acc_lstm += 1
        if df_node2vec_lstm.iloc[:, j][i]>max_ or df_node2vec_lstm.iloc[:, j][i]<min_:
            pred_node2vec_lstm += 1
            if big_change[j] :
                acc_node2vec_lstm += 1
        if df_node2vec.iloc[:, j][i]>max_ or df_node2vec.iloc[:, j][i]<min_:
            pred_node2vec += 1
            if big_change[j] :
                acc_node2vec += 1
        if df_gcn_lstm.iloc[:, j][i]>max_ or df_gcn_lstm.iloc[:, j][i]<min_ :
            pred_gcn_lstm += 1
            if big_change[j] :
                acc_gcn_lstm += 1
        if df_gat_lstm.iloc[:, j][i]>max_ or df_gat_lstm.iloc[:, j][i]<min_ :
            pred_gat_lstm += 1
            if big_change[j]:
                acc_gat_lstm += 1
        if df_gat_lstm_gat.iloc[:, j][i]>max_ or df_gat_lstm_gat.iloc[:, j][i]<min_:
            pred_gat_lstm_gat += 1
            if big_change[j]:
                acc_gat_lstm_gat += 1
        if df_gcn.iloc[:, j][i]>max_ or df_gcn.iloc[:, j][i]<min_:
            pred_gcn += 1
            if big_change[j] :
                acc_gcn += 1
        if df_gat.iloc[:, j][i]>max_ or df_gat.iloc[:, j][i]<min_:
            pred_gat += 1
            if big_change[j]:
                acc_gat += 1
        if df_node2vec_lstm_att.iloc[:, j][i]>max_ or df_node2vec_lstm_att.iloc[:, j][i]<min_:
            pred_node2vec_lstm_att += 1
            if big_change[j]:
                acc_node2vec_lstm_att += 1
        if df_gcn_lstm_att.iloc[:, j][i]>max_ or df_gcn_lstm_att.iloc[:, j][i]<min_:
            pred_gcn_lstm_att += 1
            if big_change[j]:
                acc_gcn_lstm_att += 1
        if df_gat_lstm_att.iloc[:, j][i]>max_ or df_gat_lstm_att.iloc[:, j][i]<min_:
            pred_gat_lstm_att += 1
            if big_change[j]:
                acc_gat_lstm_att += 1


        # if df_laha['tran_sum_la'][j]>max_ or df_laha['tran_sum_la'][j]<min_ or df_laha['tran_sum_la'][j+1]>max_ or df_laha['tran_sum_la'][j+1]<min_:
        #     pred_la += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_la += 1
        # if df_laha['tran_sum_ha'][j]>max_ or df_laha['tran_sum_ha'][j]<min_ or df_laha['tran_sum_ha'][j+1]>max_ or df_laha['tran_sum_ha'][j+1]<min_:
        #     pred_ha += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_ha += 1
        # if df_arima['prediction_ARIMA'][j]>max_ or df_arima['prediction_ARIMA'][j]<min_ or df_arima['prediction_ARIMA'][j+1]>max_ or df_arima['prediction_ARIMA'][j+1]<min_:
        #     pred_arima += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_arima += 1
        # if df_lstm['prediction_LSTM'][j]>max_ or df_lstm['prediction_LSTM'][j]<min_ or df_lstm['prediction_LSTM'][j+1]>max_ or df_lstm['prediction_LSTM'][j+1]<min_:
        #     pred_lstm += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_lstm += 1
        # if df_node2vec_lstm.iloc[:, j][i]>max_ or df_node2vec_lstm.iloc[:, j][i]<min_ or df_node2vec_lstm.iloc[:, j+1][i]>max_ or df_node2vec_lstm.iloc[:, j+1][i]<min_:
        #     pred_node2vec_lstm += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_node2vec_lstm += 1
        # if df_node2vec.iloc[:, j][i]>max_ or df_node2vec.iloc[:, j][i]<min_ or df_node2vec.iloc[:, j+1][i]>max_ or df_node2vec.iloc[:, j+1][i]<min_:
        #     pred_node2vec += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_node2vec += 1
        # if df_gcn_lstm.iloc[:, j][i]>max_ or df_gcn_lstm.iloc[:, j][i]<min_ or df_gcn_lstm.iloc[:, j+1][i]>max_ or df_gcn_lstm.iloc[:, j+1][i]<min_:
        #     pred_gcn_lstm += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_gcn_lstm += 1
        # if df_gat_lstm.iloc[:, j][i]>max_ or df_gat_lstm.iloc[:, j][i]<min_ or df_gat_lstm.iloc[:, j+1][i]>max_ or df_gat_lstm.iloc[:, j+1][i]<min_:
        #     pred_gat_lstm += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_gat_lstm += 1
        # if df_gat_lstm_gat.iloc[:, j][i]>max_ or df_gat_lstm_gat.iloc[:, j][i]<min_ or df_gat_lstm_gat.iloc[:, j+1][i]>max_ or df_gat_lstm_gat.iloc[:, j+1][i]<min_:
        #     pred_gat_lstm_gat += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_gat_lstm_gat += 1
        # if df_gcn.iloc[:, j][i]>max_ or df_gcn.iloc[:, j][i]<min_ or df_gcn.iloc[:, j+1][i]>max_ or df_gcn.iloc[:, j+1][i]<min_:
        #     pred_gcn += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_gcn += 1
        # if df_gat.iloc[:, j][i]>max_ or df_gat.iloc[:, j][i]<min_ or df_gat.iloc[:, j+1][i]>max_ or df_gat.iloc[:, j+1][i]<min_:
        #     pred_gat += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_gat += 1
        # if df_node2vec_lstm_att.iloc[:, j][i]>max_ or df_node2vec_lstm_att.iloc[:, j][i]<min_ or df_node2vec_lstm_att.iloc[:, j+1][i]>max_ or df_node2vec_lstm_att.iloc[:, j+1][i]<min_:
        #     pred_node2vec_lstm_att += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_node2vec_lstm_att += 1
        # if df_gcn_lstm_att.iloc[:, j][i]>max_ or df_gcn_lstm_att.iloc[:, j][i]<min_ or df_gcn_lstm_att.iloc[:, j+1][i]>max_ or df_gcn_lstm_att.iloc[:, j+1][i]<min_:
        #     pred_gcn_lstm_att += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_gcn_lstm_att += 1
        # if df_gat_lstm_att.iloc[:, j][i]>max_ or df_gat_lstm_att.iloc[:, j][i]<min_ or df_gat_lstm_att.iloc[:, j+1][i]>max_ or df_gat_lstm_att.iloc[:, j+1][i]<min_:
        #     pred_gat_lstm_att += 1
        #     if big_change[j] or big_change[j+1]:
        #         acc_gat_lstm_att += 1

rec_la = acc_la/real
rec_ha = acc_ha/real
rec_arima = acc_arima/real
rec_lstm = acc_lstm/real
rec_node2vec_lstm = acc_node2vec_lstm/real
rec_node2vec = acc_node2vec/real
rec_gcn = acc_gcn/real
rec_gat = acc_gat/real
rec_gcn_lstm = acc_gcn_lstm/real
rec_gat_lstm = acc_gat_lstm/real
rec_gat_lstm_gat = acc_gat_lstm_gat/real
rec_node2vec_lstm_att = acc_node2vec_lstm_att/real
rec_gcn_lstm_att = acc_gcn_lstm_att/real
rec_gat_lstm_att = acc_gat_lstm_att/real

try:
    pre_la = acc_la/pred_la
    f1_la = 2*pre_la*rec_la/(pre_la+rec_la)
except:
    pre_la = 0
    f1_la = 0
try:
    pre_ha = acc_ha/pred_ha
    f1_ha = 2*pre_ha*rec_ha/(pre_ha+rec_ha)
except:
    pre_ha = 0
    f1_ha = 0
try:
    pre_arima = acc_arima/pred_arima
    f1_arima = 2*pre_arima*rec_arima/(pre_arima+rec_arima)
except:
    pre_arima = 0
    f1_arima = 0
try:
    pre_lstm = acc_lstm/pred_lstm
    f1_lstm = 2*pre_lstm*rec_lstm/(pre_lstm+rec_lstm)
except:
    pre_lstm = 0
    f1_lstm = 0

try:
    pre_node2vec = acc_node2vec/pred_node2vec
    f1_node2vec = 2*pre_node2vec*rec_node2vec/(pre_node2vec+rec_node2vec)
except:
    pre_node2vec = 0
    f1_node2vec = 0

try:
    pre_node2vec_lstm = acc_node2vec_lstm/pred_node2vec_lstm
    f1_node2vec_lstm = 2*pre_node2vec_lstm*rec_node2vec_lstm/(pre_node2vec_lstm+rec_node2vec_lstm)
except:
    pre_node2vec_lstm = 0
    f1_node2vec_lstm = 0
try:
    pre_gcn_lstm = acc_gcn_lstm/pred_gcn_lstm
    f1_gcn_lstm = 2*pre_gcn_lstm*rec_gcn_lstm/(pre_gcn_lstm+rec_gcn_lstm)
except:
    pre_gcn_lstm = 0
    f1_gcn_lstm = 0
try:
    pre_gat_lstm = acc_gat_lstm/pred_gat_lstm
    f1_gat_lstm = 2*pre_gat_lstm*rec_gat_lstm/(pre_gat_lstm+rec_gat_lstm)
except:
    pre_gat_lstm = 0
    f1_gat_lstm = 0
try:
    pre_gat_lstm_gat = acc_gat_lstm_gat/pred_gat_lstm_gat
    f1_gat_lstm_gat = 2*pre_gat_lstm_gat*rec_gat_lstm_gat/(pre_gat_lstm_gat+rec_gat_lstm_gat)
except:
    pre_gat_lstm_gat = 0
    f1_gat_lstm_gat = 0
try:
    pre_gcn = acc_gcn/pred_gcn
    f1_gcn = 2*pre_gcn*rec_gcn/(pre_gcn+rec_gcn)
except:
    pre_gcn = 0
    f1_gcn = 0
try:
    pre_gat = acc_gat/pred_gat
    f1_gat = 2*pre_gat*rec_gat/(pre_gat+rec_gat)
except:
    pre_gat = 0
    f1_gat = 0
try:
    pre_node2vec_lstm_att = acc_node2vec_lstm_att/pred_node2vec_lstm_att
    f1_node2vec_lstm_att = 2*pre_node2vec_lstm_att*rec_node2vec_lstm_att/(pre_node2vec_lstm_att+rec_node2vec_lstm_att)
except:
    pre_node2vec_lstm_att = 0
    f1_node2vec_lstm_att = 0
try:
    pre_gcn_lstm_att = acc_gcn_lstm_att/pred_gcn_lstm_att
    f1_gcn_lstm_att = 2*pre_gcn_lstm_att*rec_gcn_lstm_att/(pre_gcn_lstm_att+rec_gcn_lstm_att)
except:
    pre_gcn_lstm_att = 0
    f1_gcn_lstm_att = 0
try:
    pre_gat_lstm_att = acc_gat_lstm_att/pred_gat_lstm_att
    f1_gat_lstm_att = 2*pre_gat_lstm_att*rec_gat_lstm_att/(pre_gat_lstm_att+rec_gat_lstm_att)
except:
    pre_gat_lstm_att = 0
    f1_gat_lstm_att = 0

print('pre_la: '+str(pre_la)+' rec_la: '+str(rec_la)+' f1_la: '+str(f1_la))
print('pre_ha: '+str(pre_ha)+' rec_ha: '+str(rec_ha)+' f1_ha: '+str(f1_ha))
print('pre_arima: '+str(pre_arima)+' rec_arima: '+str(rec_arima)+' f1_arima: '+str(f1_arima))
print('pre_lstm: '+str(pre_lstm)+' rec_lstm: '+str(rec_lstm)+' f1_lstm: '+str(f1_lstm))
print('pre_node2vec_lstm: '+str(pre_node2vec_lstm)+' rec_node2vec_lstm: '+str(rec_node2vec_lstm)+' f1_node2vec_lstm: '+str(f1_node2vec_lstm))
print('pre_node2vec: '+str(pre_node2vec)+' rec_node2vec: '+str(rec_node2vec)+' f1_node2vec: '+str(f1_node2vec))
print('pre_gcn_lstm: '+str(pre_gcn_lstm)+' rec_gcn_lstm: '+str(rec_gcn_lstm)+' f1_gcn_lstm: '+str(f1_gcn_lstm))
print('pre_gat_lstm: '+str(pre_gat_lstm)+' rec_gat_lstm: '+str(rec_gat_lstm)+' f1_gat_lstm: '+str(f1_gat_lstm))
print('pre_gat_lstm_gat: '+str(pre_gat_lstm_gat)+' rec_gat_lstm_gat: '+str(rec_gat_lstm_gat)+' f1_gat_lstm_gat: '+str(f1_gat_lstm_gat))
print('pre_gcn: '+str(pre_gcn)+' rec_gcn: '+str(rec_gcn)+' f1_gcn: '+str(f1_gcn))
print('pre_gat: '+str(pre_gat)+' rec_gat: '+str(rec_gat)+' f1_gat: '+str(f1_gat))
print('pre_node2vec_lstm_att: '+str(pre_node2vec_lstm_att)+' rec_node2vec_lstm_att: '+str(rec_node2vec_lstm_att)+' f1_node2vec_lstm_att: '+str(f1_node2vec_lstm_att))
print('pre_gcn_lstm_att: '+str(pre_gcn_lstm_att)+' rec_gcn_lstm_att: '+str(rec_gcn_lstm_att)+' f1_gcn_lstm_att: '+str(f1_gcn_lstm_att))
print('pre_gat_lstm_att: '+str(pre_gat_lstm_att)+' rec_gat_lstm_att: '+str(rec_gat_lstm_att)+' f1_gat_lstm_att: '+str(f1_gat_lstm_att))