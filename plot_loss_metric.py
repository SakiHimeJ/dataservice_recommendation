import matplotlib
import pandas
import numpy as np

import matplotlib.pyplot as plt

plt.style.use('ggplot')

df = pandas.read_csv('./checkpoints/test_type_view/logs/deepfm_20190621103233_epoch_52.csv')
df_total = pandas.read_csv('./checkpoints/test_type_view/logs/deepfm_20190621103235_total.csv')


loss = df['loss'].values
loss = np.log(loss)
loss_batch = df_total['loss'].values
loss_batch = np.log(loss_batch)

metric = df['metric'].values
metric_batch = df_total['metric'].values
batch_size = 8
epochs_batch = [i*batch_size for i in range(loss_batch.shape[0])]
epochs = [i for i in range(loss.shape[0])]

plt.figure(figsize=(12, 5))

plt.subplot(1,2,1)
plt.plot(epochs,loss, color='orangered', linewidth=1, linestyle="-",label='log_loss')
plt.xlabel('epochs')  # x轴标签
# plt.ylim((-1.2, -0.8))
plt.legend()  # 正常显示标题

plt.subplot(1,2,2)
plt.plot(epochs, metric, color='orange', linewidth=1, linestyle="-")
plt.plot(epochs_batch, metric_batch, color='steelblue', linewidth=1, linestyle="-",marker='o',markersize=3,label='auc')
plt.xlabel('epochs')  # x轴标签
plt.legend()  # 正常显示标题

plt.show()  # 显示图像

plt.close()
