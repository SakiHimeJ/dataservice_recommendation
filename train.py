# -*- coding:utf-8 -*-
import datetime


import data_preprocess_type_view_top5
from model import DeepFM
import pandas as pd
# from sklearn.model_select import train_test_split

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import os
import time
import matplotlib.pyplot as plt
import math

train_dict, val_dict, test_dict = data_preprocess_type_view_top5.get_fm_data()



# with GPU

'''
with torch.cuda.device(2):
    deepfm = DeepFM.DeepFM(len(train_dict['feature_sizes']), train_dict['feature_sizes'], verbose=True, use_cuda=True, weight_decay=0.0001, use_fm=True, use_ffm=False, use_deep=True).cuda()
    deepfm.fit(train_dict['index'], train_dict['value'], train_dict['label'],
           val_dict['index'], val_dict['value'], val_dict['label'], ealry_stopping=True, refit=True, save_path='./checkpoints/')
'''

# with CPU
deepfm = DeepFM.DeepFM(len(train_dict['feature_sizes']), train_dict['feature_sizes'], verbose=True, use_cuda=False, weight_decay=0.0001, use_fm=True, use_ffm=False, use_deep=True)

# 训练阶段
'''deepfm.fit(train_dict['index'], train_dict['value'], train_dict['label'],
           val_dict['index'], val_dict['value'], val_dict['label'], ealry_stopping=True, refit=True, save_path='./checkpoints/')'''


# 推断阶段
deepfm.load_state_dict(torch.load('./checkpoints/test_type_view/deepfm_20190621114313_refit.chk'))  # type_view


# 按比例设定阈值
y_pred_prob = deepfm.predict_proba(test_dict['index'], test_dict['value'])
_sort = y_pred_prob.copy()
_sort.sort(axis=0)
div_prob = int(_sort.shape[0] * (1-0.4))

div_prob = _sort[div_prob+1]
y_pred = []
for item in y_pred_prob:
    if item > div_prob:
        y_pred.append(1)
    else:
        y_pred.append(0)
y_pred = np.array(y_pred)



# y_pred = deepfm.predict(test_dict['index'], test_dict['value'])
# y_pred_prob = deepfm.predict_proba(test_dict['index'], test_dict['value'])

y_true = np.array(test_dict['label'])

fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob, pos_label=1)
plt.xlim((0, 1))
plt.ylim((0, 1))

plt.plot(fpr,tpr,color='red', linewidth=1, linestyle="-",label='ROC curve')
print(fpr.shape)
print(tpr.shape)
# np.savetxt('./roc_type_view.csv',np.vstack((fpr,tpr)).transpose(),fmt='%f',delimiter=',')

plt.show()

TP = sum(y_true * y_pred)
FP = sum(y_pred) - TP
FN = sum(y_true) - TP
TN = y_pred.shape[0] - TP - FP - FN
Precision = float(TP) / float(TP + FP)
Recall = float(TP) / float(TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
base_F1 = 2 * (0.0294 * 0.8972) / (0.0294 + 0.8972)

print('TP: %.d FP: %d' % (TP,FP))
print('FN: %.d TN: %d'% (FN,TN))
print('test Precision: %.4f Recall: %.4f' % ( Precision, Recall))
print('base Precision: %.4f Recall: %.4f' % ( 0.0294, 0.8972))
print('test F1: %.4f base F1: %.4f' % ( F1, base_F1))

Xi_test = np.array(test_dict['index']).reshape((-1, deepfm.field_size, 1))
Xv_test = np.array(test_dict['value'])
y_test = np.array(test_dict['label'])
x_test_size = Xi_test.shape[0]

test_loss, test_eval = deepfm.eval_by_batch(Xi_test, Xv_test, y_test, x_test_size)
print('*' * 50)
print( 'test loss: %.6f metric: %.6f' % ( test_loss, test_eval))
print('*' * 50)
