import os

# seen_domain = 'MNIST'
seen_domain = 'real'
# seen_domain = 'Product'
# unseen_domain = 'infograph'
# unseen_domain = 'USPS'
unseen_domain = 'clipart'
# unseen_domain = 'Real_World'
day = '4_21'
# id = 'OfficeHome_1'
# id = 'OfficeHome_only_S_0'
id = 'DomainNet_T_exp_0'
# id = 'MNIST_0'

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

base_cmd = 'CUDA_VISIBLE_DEVICES=1'
# cmd += 'CUDA_LAUNCH_BLOCKING=1 '
base_cmd += ' python main.py'
base_cmd += ' -data DomainNet'
#DomainNet OfficeHome MNIST_USPS
# cmd += ' -lr 1e-3'
base_cmd += ' -lr 2e-2'
base_cmd += ' -day '+day+' -id '+id
base_cmd += ' -model_dir '+model_dir
base_cmd += ' --num_quantzer 8'
base_cmd += ' --n_codeword 256'
base_cmd += ' --log_dir '+log_dir
base_cmd += ' --batch_size 256'
base_cmd += ' --epochs 100'
base_cmd += ' --threshold -1'
base_cmd += ' --epoch_warmup 3000000'
base_cmd += ' --num_workers 4'

cmd = base_cmd + ' -sd '+seen_domain+' -ud '+unseen_domain
os.system(cmd)

# for s in ['Product Real_World','Real_World Product','Clipart Real_World','Real_World Clipart','Art Real_World','Real_World Art']:
#     seen_domain,unseen_domain = s.split(' ')
#     cmd = base_cmd + ' -sd '+seen_domain+' -ud '+unseen_domain
#     # print(cmd)
#     os.system(cmd)