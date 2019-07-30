import numpy as np
import pandas as pd
import xgboost as xgb
import operator
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib as mpl
import data_preprocess_type_view_top5 as dp

# 解决中文乱码问题
#sans-serif就是无衬线字体，是一种通用字体族。
#常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica, 中文的幼圆、隶书等等。
mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号

# plt.style.use('ggplot')

def create_feature_map(features):
    outfile = open('./data/feat_sel/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

# dataset : Kaggle, allstate-claims-severity

if __name__ == '__main__':
    train = dp.get_dataset()

    params = {
        'min_child_weight': 100,
        'eta': 0.02,
        'colsample_bytree': 0.7,
        'max_depth': 12,
        'subsample': 0.7,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': True,
        'seed': 12
    }
    rounds = 100

    # train = train.drop(['car_no', 'mem_no'], axis=1)
    y = train['label']
    X = train.drop(['label'], 1)

    # print(X.shape)

    # feature importance
    xgtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(params, xgtrain, num_boost_round=rounds)

    features = [x for x in train.columns if x not in ['id', 'loss']]

    create_feature_map(features)

    importance = bst.get_fscore(fmap='./data/feat_sel/xgb.fmap')
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    print(importance)

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    print(df)
    df.to_csv("./data/feat_sel/feat_importance.csv", index=False)

    plt.figure()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(12, 8))
    # plt.title('XGBoost Feature Importance')
    plt.title('特征分数')
    plt.ylabel('')
    # plt.xlabel('relative importance')

    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)

    plt.show()
