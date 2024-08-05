import os
for file_name in os.listdir():
  addr = './'+file_name+'/model/*.pth'
  os.system('rm '+addr)
