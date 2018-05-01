import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split

DATA_FILE = './data_ai/fruit_data.csv'
# print(os.path.exists(DATA_FILE))

SPECIES = ['apple','orange','lemon','mandarin']

FEATURE_COl = ['mass','width','height','color_score']

def get_predict(test_sample_feat,train_data):
    dis_list = []
    for idx,row in train_data.iterrows():
        train_sample_feat = row[FEATURE_COl].values

        dis = euclidean(test_sample_feat,train_sample_feat)
        dis_list.append(dis)

    pos = np.argmin(dis_list)
    pred_label = train_data.iloc[pos]['fruit_name']
    return pred_label

def classifier():
    fruit_data = pd.read_csv(DATA_FILE)

    train_data, test_data = train_test_split(fruit_data,test_size=1/5,random_state=20)

    acc_count = 0
    for idx,row in test_data.iterrows():
        test_sample_feat = row[FEATURE_COl].values
        pred_label = get_predict(test_sample_feat,train_data)
        true_label = row['fruit_name']

        print('data{}, true:{}, predict:{}'.format(idx,true_label,pred_label))
        if pred_label==true_label:
            acc_count+=1
    print('Accuracy: {:.2f}%'.format(acc_count/test_data.shape[0]*100))



classifier()