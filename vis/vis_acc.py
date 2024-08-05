import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib
 

# data = np.load('/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/acc_dist_JS_65.npy')
# data = np.load('/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/acc_dist_L2.npy')
# file_name = 'select_JS'
# data = np.load('/home/zhangzhibin/data/UDA/CDTrans-master/UDA_Q/'+file_name+'.npy')
dataset = 'DomainNet'
# OfficeHome DomainNet
data = np.load('prob_'+dataset+'.npy')
n_data = data.shape[0]
print(np.mean(data[:int(n_data*0.05)]))
print(np.std(data[:int(n_data*0.05)]))
print(data[int(n_data*0.5)])
print('======')
mean = np.mean(data)
std = np.std(data)
sum = np.sum( (data > mean+std).astype('int') )
print(mean)
print(std)
# print(n_data)
# print(sum)

data = (data*100)
np.round_(data)
data = data.astype('int')
print(data)

data = np.bincount(data,minlength=100)
print(data)

x = (np.array( range(n_data) )+1)/n_data
# x_ticks = (np.array( range(n_data) )+1)/n_data
y = data

# plt.ylim(top=400)
plt.xlabel("rate")
plt.ylabel("select times")
# plt.plot(x,y) 
# plt.xticks(x, x_ticks)
plt.bar(range(data.shape[0]), data)
plt.savefig('prob_'+dataset+'_bin.png')