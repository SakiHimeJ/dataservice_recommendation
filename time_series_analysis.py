import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.dates as dates
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF

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
nrows=4
fig,ax=plt.subplots(figsize=(ncols*4,nrows*2.5), ncols=ncols,nrows=nrows)

idx=df_entry.loc['2018-01-01':'2019-01-01'].index
value=df_entry.loc['2018-01-01':'2019-01-01']['entry_count']

# 阿道夫统计量：adf: (-2.6039596171535924, 0.09218165356068125, 14, 260, {'1%': -3.4557539868570775, '5%': -2.8727214497041422, '10%': -2.572728476331361}, 4921.65497015057)

'''
第一个是adt检验的结果，也就是t统计量的值。
第二个是t统计量的P值。
第三个是计算过程中用到的延迟阶数。
第四个是用于ADF回归和计算的观测值的个数。
第五个是配合第一个一起看的，是在99%，95%，90%置信区间下的临界的ADF检验的值。如果第一个值比第五个值小证明平稳，反正证明不平稳。
'''
adf = ADF(value)
print('adf:',adf)
ax[0][0].plot_date(idx, value, 'r-',label='点击量')
value=value.diff(2)
adf = ADF(value.dropna())
print('adf_diff:',adf)
ax[1][0].plot_date(idx, value, 'r-',label='点击量（差分）')
fig = sm.graphics.tsa.plot_acf(value.dropna(),lags=40,ax=ax[2][0])
ax[2][0].set_title('自相关系数')
fig = sm.graphics.tsa.plot_pacf(value.dropna(),lags=40,ax=ax[3][0])
ax[3][0].set_title('偏自相关系数')

idx=df_order.loc['2018-01-01':'2019-01-01'].index
value=df_order.loc['2018-01-01':'2019-01-01']['req_count']

adf = ADF(value)
print('adf:',adf)
ax[0][1].plot_date(idx, value, 'm-',label='请求订单')
value=value.diff(1)
adf = ADF(value.dropna())
print('adf_diff:',adf)
ax[1][1].plot_date(idx, value, 'm-',label='请求订单（差分）')
fig = sm.graphics.tsa.plot_acf(value.dropna(),lags=40,ax=ax[2][1])
ax[2][1].set_title('自相关系数')
fig = sm.graphics.tsa.plot_pacf(value.dropna(),lags=40,ax=ax[3][1])
ax[3][1].set_title('偏自相关系数')


idx=df_order.loc['2018-01-01':'2019-01-01'].index
value=df_order.loc['2018-01-01':'2019-01-01']['suc_count']

adf = ADF(value)
print('adf:',adf)
ax[0][2].plot_date(idx, value, 'c-',label='成功订单')
value=value.diff(1)
adf = ADF(value.dropna())
print('adf_diff:',adf)
ax[1][2].plot_date(idx, value, 'c-',label='成功订单（差分）')
fig = sm.graphics.tsa.plot_acf(value.dropna(),lags=40,ax=ax[2][2])
ax[2][2].set_title('自相关系数')
fig = sm.graphics.tsa.plot_pacf(value.dropna(),lags=40,ax=ax[3][2])
ax[3][2].set_title('偏自相关系数')

#如果坐标轴上面的tick过于密集
# fig.autofmt_xdate()#自动调整xtick的间距

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


