import os

seen_domain = 'real'
# unseen_domain = 'infograph'
unseen_domain = 'clipart'
day = '3_24'
id = 'L2_0'

day_dir = 'logs/'+day+'/'
dir = day_dir+'_'.join((day,id,seen_domain,unseen_domain))
print(dir)
model_dir = dir+'/model'
code_dir = dir+'/code'

os.system('mkdir '+day_dir)
os.system('mkdir '+dir)
os.system('mkdir '+model_dir)
os.system('mkdir '+code_dir)

os.system('cp *.py '+code_dir)

cmd = 'CUDA_VISIBLE_DEVICES=2 python main.py'
cmd += ' -sd '+seen_domain+' -ud '+unseen_domain
cmd += ' -day '+day+' -id '+id
cmd += ' -model_dir '+model_dir
cmd += ' --num_quantzer 4'
print(cmd)

os.system(cmd)