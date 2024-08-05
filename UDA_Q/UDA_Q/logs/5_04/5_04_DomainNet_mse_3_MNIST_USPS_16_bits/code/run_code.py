from builtins import all
import os

def cmd_from_config(file_path):
    from configparser import ConfigParser
    config = ConfigParser()
    config.read(file_path, encoding='UTF-8')
    s = ''
    for sec in config.sections():
        for (key,val) in config.items(sec):
            s += ' --'+key+' '+val
    return s

def run_cmd(seen_domain,unseen_domain,num_quantizer,per_bit,suff=''):
    n_codeword = 2**per_bit
    day = '5_04'
    # id = 'OfficeHome_fin_only_S_0'
    # id = 'DomainNet_fin_0'
    # id = 'DomainNet_clip_0'
    # id = 'DomainNet_only_S_0'
    id = 'DomainNet_mse_3'
    # id = 'MNIST_exp_lr_rewrite_0'
    # id = 'MNIST_fin_1'
    # id = 'MNIST_vgg_bn_all_1'
    # id = 'test'

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
    cmd += ' -data MNIST_USPS'
    #DomainNet OfficeHome MNIST_USPS
    
    cmd += ' -day '+day+' -id '+id
    cmd += ' -model_dir '+model_dir
    cmd += ' --log_dir '+log_dir
    cmd += ' --num_workers 4'
    cmd += ' -sd '+seen_domain+' -ud '+unseen_domain
    cmd += ' --num_quantzer '+str(num_quantizer)
    cmd += ' --n_codeword '+str(n_codeword)
    cmd += cmd_from_config('./configs/MNIST_USPS.cfg')
    
    dim = num_quantizer*16
    cmd += ' --dim '+str(dim)
    # os.system(cmd)
    print(cmd)

# MNIST: lr= 2e-3 bs=512 sgd
# OfficeHome: lr=2e-2 bs=64 sgd dim = 8*16 = 128 cls_gamma 0.5
# seen_domain = 'MNIST'
# seen_domain = 'real'
# seen_domain = 'Product'
# unseen_domain = 'infograph'
# unseen_domain = 'USPS'
# unseen_domain = 'clipart'
# unseen_domain = 'Real_World'
num_quantizer = 8
per_bit = 8
# run_cmd('real','clipart',num_quantizer,per_bit)
# for s in ['Product Real_World','Real_World Product','Clipart Real_World','Real_World Clipart','Art Real_World','Real_World Art']:
# for s in ['real clipart']:
# for s in ['real quickdraw','real infograph']:
# for s in ['Product Real_World']:
    # seen_domain,unseen_domain = s.split(' ')
    # run_cmd(seen_domain,unseen_domain,num_quantizer,per_bit)


# for (num_quantizer,per_bit) in [ (2,8),(4,8),(8,6),(8,8),(16,6),(16,8)]:
for (num_quantizer,per_bit) in [ (2,8)]:
    all_bits = num_quantizer*per_bit
    print(all_bits)
    run_cmd('MNIST','USPS',num_quantizer,per_bit,'_'+str(all_bits)+'_bits')