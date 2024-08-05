import os

# seen_domain = 'real'
# seen_domain = 'clipart'
seen_domain = 'CIFAR10'
# unseen_domain = 'infograph'
# unseen_domain = 'real'
unseen_domain = 'CIFAR10'
day = '4_13'
id = 'CIFAR_2'

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

cmd = 'CUDA_VISIBLE_DEVICES=1'
# cmd += 'CUDA_LAUNCH_BLOCKING=1 '
cmd += ' python main.py'
cmd += ' -data CIFAR10'
cmd += ' -lr 2e-2'
# cmd += ' -sd '+seen_domain+' -ud '+unseen_domain
cmd += ' -day '+day+' -id '+id
cmd += ' -model_dir '+model_dir
cmd += ' --num_quantzer 4'
cmd += ' --log_dir '+log_dir
cmd += ' --batch_size 512'
cmd += ' --epochs 100'
cmd += ' --threshold -1'
cmd += ' --epoch_warmup 1000'
# cmd += ' > '+log_dir+' &'
os.system(cmd)