# 使用ARIMA模型预测，这里用的还是12个节点对

from statsmodels.tsa.arima_model import ARIMA
# import pmdarima as pm
# from pyramid.arima import auto_arima
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import math



name = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\name_node_pairs_2_quchong_with12_without_notran.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

mse = 0

for i in range(len(name_node_pairs)):

    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\temporal link prediction_ARIMA\\'+name_node_pairs[i]+'.csv','w',newline='')
    csvwriter = csv.writer(file)
    csvwriter.writerow(['t', 'tran_sum_real', 'prediction_ARIMA', 'difference_ARIMA'])
    for j in range(2):
        csvwriter.writerow([j+3, 0., 0., 0.])
    file.close()

# 读取每个节点对的交易记录
num = 0
for i in range(len(name_node_pairs)):
    print(i)
    print(str(i / len(name_node_pairs) * 100) + '%')
    # print(name_node_pairs[i])

    file = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\temporal link features_5_7days_739\\' + name_node_pairs[i] + '_temp_link_ft.csv')
    df = pd.read_csv(file)

    new_data = pd.DataFrame(df, columns=['tran_sum'])
    train_ = new_data[0:3]
    # print('type(new_data)')
    # print(type(new_data))
    dataset = new_data.values
    # print('type(dataset)')
    # print((type(dataset)))

    # plt.figure
    # plt.plot(new_data,'x--')
    # plt.show()

    # creating train and test sets
    train = dataset[0:3]
    valid = dataset[3:5]

    # ======================== model  fit ================================

    # model = pm.auto_arima(train_, start_p=1, start_q=1,
    #                       information_criterion='aic',
    #                       test='adf',  # use adftest to find optimal 'd'
    #                       max_p=3, max_q=3,  # maximum p and q
    #                       m=1,  # frequency of series
    #                       d=None,  # let model determine 'd'
    #                       seasonal=False,  # No Seasonality
    #                       start_P=0,
    #                       D=0,
    #                       trace=True,
    #                       error_action='ignore',
    #                       suppress_warnings=True,
    #                       stepwise=True)
    # print(model.summary())

    # model = auto_arima(train_, trace=True, error_action='ignore', suppress_warnings=True)
    model = ARIMA(train_, order=(1,0,0))
    try:
        fitted = model.fit(disp=-1)

        # =====================预测 =======================================

        # Forecast
        n_periods = 2
        fc, se, conf = fitted.forecast(n_periods)
        # print('type(fc)')
        # print((fc))
        tran_sum = fc
        # ===================================================================
    except:
        print('exception')
        print(train_)
        tran_sum = [train_.values.mean() for i in range(2)]
    # rms = np.sqrt(np.mean(np.power((valid-tran_sum), 2)))
    rms = (np.mean(np.power((valid - tran_sum), 2)))
    # print(rms)
    mse = mse+rms

    file2 = open('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\temporal link prediction_ARIMA\\' + name_node_pairs[i] + '.csv')
    df_2 = pd.read_csv(file2)
    for j in range(2):
        df_2['prediction_ARIMA'][j] = tran_sum[j]
        df_2['tran_sum_real'][j] = df['tran_sum'][j+3]
        df_2['difference_ARIMA'][j] = df_2['prediction_ARIMA'][j]-df_2['tran_sum_real'][j]
    df_2.to_csv('E:\\exchange_0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be_12\\temporal link prediction_ARIMA\\' + name_node_pairs[i] + '.csv', index=False)

    # train = new_data[:3]
    # valid = new_data[3:]
    # valid['Predictions'] = fc
    # plt.plot(train['tran_sum'],'x--')
    # plt.plot(valid[['tran_sum', 'Predictions']],'x--')
    # plt.legend(['tran_sum', 'tran_sum', 'Predictions'])
    # plt.xlabel('time')
    # plt.ylabel('transaction value')
    # plt.show()

rmse = np.sqrt(mse/len(name_node_pairs))
print(rmse)