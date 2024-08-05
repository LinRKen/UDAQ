import os

seen_domain = 'Real_World'
# seen_domain = 'clipart'
# seen_domain = 'ImageNet'
# unseen_domain = 'infograph'
# unseen_domain = 'real'
unseen_domain = 'Product'
# unseen_domain = 'ImageNet'
day = '4_15'
id = 'OfficeHome_0'

day_dir = 'logs/'+day+'/'
dir = day_dir+'_'.join((day,id,seen_domain,unseen_domain))
print(dir)
model_dir = dir+'/model'
code_dir = dir+'/code'
log_dir = dir+'/logs.txt'

os.system('mkdir '+day_dir)
os.system('mkdir '+dir)
os.system('mkdir '+model_dir)
os.system('mkdir '+code_dir)
os.system('mkdir '+code_dir+'/root/')

os.system('cp *.py '+code_dir)
os.system('cp ../*.py '+code_dir+'/root/')

cmd = 'CUDA_VISIBLE_DEVICES=3'
# cmd += 'CUDA_LAUNCH_BLOCKING=1 '
cmd += ' python main.py'
cmd += ' -data OfficeHome'
# ImageNet CIFAR10
cmd += ' -lr 2e-2'
cmd += ' -sd '+seen_domain+' -ud '+unseen_domain
cmd += ' -day '+day+' -id '+id
cmd += ' -model_dir '+model_dir
cmd += ' --num_quantzer 8'
cmd += ' --n_codeword 256'
cmd += ' --log_dir '+log_dir
cmd += ' --batch_size 512'
cmd += ' --epochs 1000'
cmd += ' --threshold -1'
cmd += ' --epoch_warmup 1000'
# cmd += ' > '+log_dir+' &'
print(cmd)
os.system(cmd)