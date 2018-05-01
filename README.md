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

# 2. Code

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
