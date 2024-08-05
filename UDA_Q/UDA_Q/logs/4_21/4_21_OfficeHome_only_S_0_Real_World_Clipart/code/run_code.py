import os

def run_cmd(seen_domain,unseen_domain):
    day = '4_21'
    # id = 'OfficeHome_1'
    id = 'OfficeHome_only_S_0'
    # id = 'DomainNet_T_exp_0'
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

    cmd = 'CUDA_VISIBLE_DEVICES=0'
    # cmd += 'CUDA_LAUNCH_BLOCKING=1 '
    cmd += ' python main.py'
    cmd += ' -data DomainNet'
    #DomainNet OfficeHome MNIST_USPS
    # cmd += ' -lr 1e-3'
    cmd += ' -lr 2e-2'
    cmd += ' -day '+day+' -id '+id
    cmd += ' -model_dir '+model_dir
    cmd += ' --num_quantzer 8'
    cmd += ' --n_codeword 256'
    cmd += ' --log_dir '+log_dir
    cmd += ' --batch_size 1024'
    cmd += ' --epochs 100'
    cmd += ' --threshold -1'
    cmd += ' --epoch_warmup 30'
    cmd += ' --num_workers 4'

    cmd = cmd + ' -sd '+seen_domain+' -ud '+unseen_domain
    os.system(cmd)

# seen_domain = 'MNIST'
# seen_domain = 'real'
# seen_domain = 'Product'
# unseen_domain = 'infograph'
# unseen_domain = 'USPS'
# unseen_domain = 'clipart'
# unseen_domain = 'Real_World'
run_cmd('real','clipart')
# for s in ['Product Real_World','Real_World Product','Clipart Real_World','Real_World Clipart','Art Real_World','Real_World Art']:
#     seen_domain,unseen_domain = s.split(' ')
#     run_cmd(seen_domain,unseen_domain)