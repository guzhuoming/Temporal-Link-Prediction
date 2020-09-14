import pandas as pd
import numpy as np
import csv
import math

name = open('./data/name_node_pairs_大额异动.csv')
df_name_node_pairs = pd.read_csv(name)
name_node_pairs = df_name_node_pairs['name_node_pairs']

file = open('./data/accounts_label_phish-hack.csv')
df_phish = pd.read_csv(file)
phish_add = df_phish['Address'].values.tolist()

for i in range(len(name_node_pairs)):
    for j in phish_add:
        if j==name_node_pairs[i]:
            print(name_node_pairs[i])

