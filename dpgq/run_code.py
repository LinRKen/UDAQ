from builtins import all
import os

def cmd_from_config(file_path):
    from configparser import ConfigParser
    config = ConfigParser()
    config.optionxform = lambda option: option
    config.read(file_path, encoding='UTF-8')
    s = ''
    for sec in config.sections():
        for (key,val) in config.items(sec):
            s += ' --'+key+' '+val
    return s

def run_cmd(dataset,seen_domain,unseen_domain,num_quantizer,per_bit,suff=''):
    n_codeword = 2**per_bit
    day = '5_08'
    id = 'DomainNet_3'
    

    day_dir = 'logs/'+day+'/'
    dir = day_dir+'_'.join((day,id,seen_domain,unseen_domain))
    dir += suff
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
    cmd += ' -data '+dataset
    #DomainNet OfficeHome MNIST_USPS
    
    cmd += ' -day '+day+' -id '+id
    cmd += ' -model_dir '+model_dir
    cmd += ' --log_dir '+log_dir
    cmd += ' --num_workers 4'
    cmd += ' -sd '+seen_domain+' -ud '+unseen_domain
    cmd += ' --num_quantzer '+str(num_quantizer)
    cmd += ' --n_codeword '+str(n_codeword)
    
    cmd += ' --lr 2e-2'
    cmd += ' --weight_decay 0'
    cmd += ' --Q_lr 1e-4'
    cmd += ' --Q_weight_decay 0'
    cmd += ' --opt sgd'
    cmd += ' --batch_size 64'
    cmd += ' --epochs 100'
    cmd += ' --warm_epoch 50'    
    cmd += ' --dim 300'

    print(cmd)
    os.system(cmd)

num_quantizer = 8
per_bit = 8
for s in ['real clipart','real infograph','real painting','real sketch','real quickdraw']:
    seen_domain,unseen_domain = s.split(' ')
    dataset = 'DomainNet'
    run_cmd(dataset,seen_domain,unseen_domain,num_quantizer,per_bit)