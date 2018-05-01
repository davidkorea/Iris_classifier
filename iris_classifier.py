import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import numpy as np
import os

import ai_utils

DATA_FILE = './data_ai/Iris.csv'
# print(os.path.exists(DATA_FILE))

SPECIES = ['Iris-setosa',
           'Iris-versicolor',
           'Iris-virginica']

FEATURE_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

def get_pred_label(test_sample_feat,train_data):
    dis_list = []
    for idx,row in train_data.iterrows():
        train_sample_feat = row[FEATURE_COLS].values
        dis = euclidean(test_sample_feat,train_sample_feat)
        dis_list.append(dis)

    pos = np.argmin(dis_list)
    pred_label = train_data.iloc[pos]['Species']
    return pred_label


def main():
    iris_data = pd.read_csv(DATA_FILE,index_col='Id')

    # EDA
    ai_utils.do_eda_plot_for_iris(iris_data)

    # split train / tet
    train_data,test_data = train_test_split(iris_data, test_size=1/3,random_state=10)

    acc_count = 0
    for idx,row in test_data.iterrows():
        test_sample_feat = row[FEATURE_COLS].values

        # predict value
        pred_label = get_pred_label(test_sample_feat,train_data)

        #true label
        true_label = row['Species']

        print('Sample{}, true:{}, predict:{}'.format(idx,true_label,pred_label))

        if pred_label == true_label:
            acc_count+=1
    print('Accuracy:{}'.format(acc_count/test_data.shape[0]*100))


main()