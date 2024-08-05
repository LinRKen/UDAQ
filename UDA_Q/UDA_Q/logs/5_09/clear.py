import os
for file_name in os.listdir():
  addr = '/mnt/hdd1/zhangzhibin/UDA/CDTrans-master/UDA_Q/logs/5_09/'+file_name+'/model/*.pth'
  os.system('rm '+addr)
