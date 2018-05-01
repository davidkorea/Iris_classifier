# Iris_classifier

# 0. Subject
1. 探索性数据分析EDA
2. 机器学习基础概念
3. 实现简单分类器 - 通过最近距离分类
 
# 1. Basic
 
 机器学习训练样本中的特征
![](https://github.com/davidkorea/Iris_classifier/blob/master/images/basic1.png)

![](https://github.com/davidkorea/Iris_classifier/blob/master/images/basic2.png)

![](https://github.com/davidkorea/Iris_classifier/blob/master/images/basic3.png)

![](https://github.com/davidkorea/Iris_classifier/blob/master/images/basic4.png)

![](https://github.com/davidkorea/Iris_classifier/blob/master/images/basic5.png)

![](https://github.com/davidkorea/Iris_classifier/blob/master/images/basic6.png)

# 2. 코드 요약

0. preparation
```php
DATA_PATH = ''
SPECIES = ['apple','orange','lemon']
FEATURE_COL = ['mass','width','heught','color_score']
```
1. read data file
```php
data_file = pandas.read_csv(DATA_PATH, index_col='') #index_col could be blanck
```
2. split data file to Train & Test data
```php
train_data, test_data = train_test_split(data_file, test_size=1/5, random_state=20) 
 # 1/5 of origin data will be set as test data
```
3. get each item's feature in the test sample
```php
for idx, row in test_data.iterrows():
  test_sample_feat = row[FEATURE_COL].values
  pred_label = get_predict(test_sample_feat, train_data)
```
test_data.iterrows() DEBUG:
```php
In[2]:est_data.iterrows
Out[2]: 
<bound method DataFrame.iterrows of    fruit_name  mass  width  height  color_score
4    mandarin    84    6.0     4.6         0.79
23      apple   170    7.6     7.9         0.88
17      apple   168    7.5     7.6         0.73
56      lemon   116    5.9     8.1         0.73
37     orange   154    7.3     7.3         0.79
12      apple   154    7.0     7.1         0.88
52      lemon   118    5.9     8.0         0.72
5    mandarin    80    5.8     4.3         0.77
24     orange   342    9.0     9.4         0.75
36     orange   160    7.1     7.6         0.76
44      lemon   200    7.3    10.5         0.72
2       apple   176    7.4     7.2         0.60>

```

![](https://github.com/davidkorea/Iris_classifier/blob/master/images/color_score.jpg)

4. get_predict()
```php
def get_predict(test_sample_feat, train_data):
  dis_list = []
  for idx, row in train_data.iterrows():
    train_sample_feat = row[FEATURE_COL].values
    dis = euclidean(test_sample_feat, train_sample_feat)
    dis_list.append(dis)
  pos = numpy.argmin(dis_list)
  pred_label = train_data.iloc[pos]['fruit_name']
  return pred_label
```

5. accuracy
```php
for idx, row in test_data.iterrows():
  test_sample_feat = row[FEATURE_COL].values
  pred_label = get_predict(test_sample_feat, train_data)
  <!-- SAME AS step-3 above--!>
  true_label = row['fruit_name']
  
  print('DATA:{}, true: {}, predict :{}'.format(idx, true_label, pred_label))
  
  acc_count = 0
  if pred_label == true_label:
    acc_count += 1
print('Accuracy: {:.2f}%'.format(acc_count/test_data.shape[0]*100))
```
**Result**
```
data4, true:mandarin, predict:mandarin
data23, true:apple, predict:apple
data17, true:apple, predict:apple
data56, true:lemon, predict:lemon
data37, true:orange, predict:orange
data12, true:apple, predict:orange
data52, true:lemon, predict:lemon
data5, true:mandarin, predict:mandarin
data24, true:orange, predict:orange
data36, true:orange, predict:orange
data44, true:lemon, predict:lemon
data2, true:apple, predict:apple
Accuracy: 91.67%
```




# 3. 코드 분석

```python
"""
    任务：鸢尾花识别
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import numpy as np
import ai_utils //EDA script

DATA_FILE = './data_ai/Iris.csv'

SPECIES = ['Iris-setosa',       # 山鸢尾
           'Iris-versicolor',   # 变色鸢尾
           'Iris-virginica'     # 维吉尼亚鸢尾
           ]

# 使用的特征列
FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


def get_pred_label(test_sample_feat, train_data):
    """
        “近朱者赤” 找最近距离的训练样本，取其标签作为预测样本的标签
    """
    dis_list = []

    for idx, row in train_data.iterrows():
        # 训练样本特征
        train_sample_feat = row[FEAT_COLS].values //训练集中的一个数据
 
        # 计算距离
        dis = euclidean(test_sample_feat, train_sample_feat) //测试集的一个数据 与 训练集的一个数据 计算距离
        dis_list.append(dis) //测试集的一个数据 与 所有训练集数据 距离的 所有值
 
    # 最小距离对应的位置
    pos = np.argmin(dis_list)
    pred_label = train_data.iloc[pos]['Species']
    return pred_label


def main():
    """
        主函数
    """
    # 读取数据集
    iris_data = pd.read_csv(DATA_FILE, index_col='Id') //使用pandas读取数据文件
    # EDA
    ai_utils.do_eda_plot_for_iris(iris_data) //使用小脚本查看各个特征的分布图
 、
    # 划分数据集
    train_data, test_data = train_test_split(iris_data, test_size=1/3, random_state=10) //使用已有工具包划分训练/测试数据集
    
    # 预测对的个数
    acc_count = 0

    # 分类器
    for idx, row in test_data.iterrows():  //遍历测试集中的每个数据
        # 测试样本特征
        test_sample_feat = row[FEAT_COLS].values //测试集中的一行数据
 
        # 预测值
        pred_label = get_pred_label(test_sample_feat, train_data) //测试集的一个数据特征 与 训练集的 每一行数据 特征计算距离
        
        # 真实值
        true_label = row['Species']
        print('样本{}的真实标签{}，预测标签{}'.format(idx, true_label, pred_label))

        if true_label == pred_label:
            acc_count += 1

    # 准确率
    accuracy = acc_count / test_data.shape[0]
    print('预测准确率{:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    main()
```
