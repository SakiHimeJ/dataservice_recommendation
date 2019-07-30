import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as dates
import matplotlib as mpl
import statsmodels.api as sm
import statsmodels.tsa.stattools as st

# 解决中文乱码问题
#sans-serif就是无衬线字体，是一种通用字体族。
#常见的无衬线字体有 Trebuchet MS, Tahoma, Verdana, Arial, Helvetica, 中文的幼圆、隶书等等。
mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号

# plt.style.use('ggplot')

df_entry=pd.read_csv('data/2018_entry.csv',delimiter='\t', index_col=0,parse_dates=True)
df_order=pd.read_csv('data/2018_order.csv',delimiter='\t', index_col=0,parse_dates=True)

# df_entry = df_entry.sort_values(by=['entrydate'])
# print(df_entry)

ncols=3
nrows=2
fig,ax=plt.subplots(figsize=(ncols*6,nrows*4), ncols=ncols,nrows=nrows)

# model 1
idx_train=df_entry.loc['2018-04-01':'2018-12-01'].index
value_train=df_entry.loc['2018-04-01':'2018-12-01']['entry_count']

idx_test=df_entry.loc['2018-12-01':'2019-01-01'].index
value_test=df_entry.loc['2018-12-01':'2019-01-01']['entry_count']

# order = st.arma_order_select_ic(value_train.diff(1).dropna(),max_ar=7,max_ma=7,ic=['aic','bic','hqic'])
# print(order.bic_min_order) # (4, 6)

arma_mod1 = sm.tsa.ARMA(value_train.diff(1).dropna(),(4,6)).fit()

predict_sunspots = arma_mod1.predict()

idx_pred = pd.date_range('2018-04-02','2018-12-01', normalize=True)
ax[0][0].plot_date(idx_train, value_train.diff(1), 'r-',label='点击量（差分）')
ax[0][0].plot_date(idx_pred, predict_sunspots, 'y-',label='拟合序列')

# 预测
idx_pred = pd.date_range('2018-12-02','2019-01-01', normalize=True)
predict_sunspots_test = arma_mod1.predict(start='2018-12-02',end='2019-01-01')
ax[0][0].plot_date(idx_pred, predict_sunspots_test, 'y-')

idx_pred = pd.date_range('2018-04-02','2018-12-01', normalize=True)
ax[1][0].plot_date(idx_train, value_train, 'r-',label='点击量')
ax[1][0].plot_date(idx_pred, predict_sunspots.add(value_train.shift(1).dropna()), 'y-',label='拟合序列')

# 预测
idx_pred = pd.date_range('2018-12-02','2019-01-01', normalize=True)
predict_sunspots_test = arma_mod1.predict(start='2018-12-02',end='2019-01-01')
ax[1][0].plot_date(idx_pred, predict_sunspots_test.add(value_test.shift(1).dropna()), 'y-')

# model 2
idx_train=df_order.loc['2018-04-01':'2018-12-01'].index
value_train=df_order.loc['2018-04-01':'2018-12-01']['req_count']

idx_test=df_order.loc['2018-12-01':'2019-01-01'].index
value_test=df_order.loc['2018-12-01':'2019-01-01']['req_count']

# 定阶,输出最优阶数
# order = st.arma_order_select_ic(value_train.diff(1).dropna(),max_ar=7,max_ma=7,ic=['aic','bic','hqic'])
# print(order.bic_min_order) # (4, 6)

arma_mod1 = sm.tsa.ARMA(value_train.diff(1).dropna(),(4,6)).fit()

predict_sunspots = arma_mod1.predict()

# print(predict_sunspots)
idx_pred = pd.date_range('2018-04-02','2018-12-01', normalize=True)
ax[0][1].plot_date(idx_train, value_train.diff(1), 'm-',label='请求订单（差分）')
ax[0][1].plot_date(idx_pred, predict_sunspots, 'y-',label='拟合序列')

# 预测
idx_pred = pd.date_range('2018-12-02','2019-01-01', normalize=True)
predict_sunspots_test = arma_mod1.predict(start='2018-12-02',end='2019-01-01')
ax[0][1].plot_date(idx_pred, predict_sunspots_test, 'y-')

idx_pred = pd.date_range('2018-04-02','2018-12-01', normalize=True)
ax[1][1].plot_date(idx_train, value_train, 'm-',label='请求订单')
ax[1][1].plot_date(idx_pred, predict_sunspots.add(value_train.shift(1).dropna()), 'y-',label='拟合序列')

# 预测
idx_pred = pd.date_range('2018-12-02','2019-01-01', normalize=True)
predict_sunspots_test = arma_mod1.predict(start='2018-12-02',end='2019-01-01')
ax[1][1].plot_date(idx_pred, predict_sunspots_test.add(value_test.shift(1).dropna()), 'y-')


# model 3
idx_train=df_order.loc['2018-04-01':'2018-12-01'].index
value_train=df_order.loc['2018-04-01':'2018-12-01']['suc_count']

idx_test=df_order.loc['2018-12-01':'2019-01-01'].index
value_test=df_order.loc['2018-12-01':'2019-01-01']['suc_count']

# order = st.arma_order_select_ic(value_train.diff(1).dropna(),max_ar=7,max_ma=7,ic=['aic','bic','hqic'])
# print(order.bic_min_order) # (4, 5)

arma_mod1 = sm.tsa.ARMA(value_train.diff(2).dropna(),(4,5)).fit()

predict_sunspots = arma_mod1.predict()

idx_pred = pd.date_range('2018-04-03','2018-12-01', normalize=True)
ax[0][2].plot_date(idx_train, value_train.diff(2), 'c-',label='成功订单（差分）')
ax[0][2].plot_date(idx_pred, predict_sunspots, 'y-',label='拟合序列')

# 预测
idx_pred = pd.date_range('2018-12-02','2019-01-01', normalize=True)
predict_sunspots_test = arma_mod1.predict(start='2018-12-02',end='2019-01-01')
ax[0][2].plot_date(idx_pred, predict_sunspots_test, 'y-')

idx_pred = pd.date_range('2018-04-03','2018-12-01', normalize=True)
ax[1][2].plot_date(idx_train, value_train, 'c-',label='成功订单')
ax[1][2].plot_date(idx_pred, predict_sunspots.add(value_train.shift(2).dropna()), 'y-',label='拟合序列')

# 预测
idx_pred = pd.date_range('2018-12-02','2019-01-01', normalize=True)
predict_sunspots_test = arma_mod1.predict(start='2018-12-02',end='2019-01-01')
ax[1][2].plot_date(idx_pred, predict_sunspots_test.add(value_test.shift(2).dropna()), 'y-')

#网格
for i in range(nrows):
    for j in range(ncols):
        if i < 2:
            ax[i][j].xaxis.grid(True)
            ax[i][j].yaxis.grid(True)

            #设置日期为每个月
            #Location也就是以什么样的频率
            ax[i][j].xaxis.set_major_locator(dates.MonthLocator())
            #Format坐标轴展示的样式
            ax[i][j].xaxis.set_major_formatter(dates.DateFormatter('%b-%Y'))
            ax[i][j].legend()
            for label in ax[i][j].get_xticklabels():
                label.set_ha('right')
                label.set_rotation(30.)

plt.tight_layout()

plt.show()






