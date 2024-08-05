from builtins import all
import os

def run_cmd(seen_domain,unseen_domain,num_quantizer,per_bit,suff=''):
    n_codeword = 2**per_bit
    day = '4_22'
    # id = 'OfficeHome_1'
    # id = 'OfficeHome_only_S_0'
    # id = 'OfficeHome_exp_T_0'
    # id = 'DomainNet_exp_1'
    id = 'MNIST_no_crop_exp_cos_0'
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

    cmd = 'CUDA_VISIBLE_DEVICES=0'
    # cmd += 'CUDA_LAUNCH_BLOCKING=1 '
    cmd += ' python main.py'
    cmd += ' -data MNIST_USPS'
    #DomainNet OfficeHome MNIST_USPS
    # cmd += ' -lr 1e-3'
    cmd += ' -lr 2e-2'
    cmd += ' -day '+day+' -id '+id
    cmd += ' -model_dir '+model_dir
    # cmd += ' --num_quantzer 8'
    cmd += ' --log_dir '+log_dir
    cmd += ' --batch_size 512'
    cmd += ' --epochs 120'
    cmd += ' --threshold -1'
    cmd += ' --epoch_warmup 30'
    cmd += ' --num_workers 4'

    cmd += ' -sd '+seen_domain+' -ud '+unseen_domain
    cmd += ' --num_quantzer '+str(num_quantizer)
    cmd += ' --n_codeword '+str(n_codeword)
    os.system(cmd)

# seen_domain = 'MNIST'
# seen_domain = 'real'
# seen_domain = 'Product'
# unseen_domain = 'infograph'
# unseen_domain = 'USPS'
# unseen_domain = 'clipart'
# unseen_domain = 'Real_World'
# run_cmd('real','clipart')
# for s in ['Product Real_World','Real_World Product','Clipart Real_World','Real_World Clipart','Art Real_World','Real_World Art']:
#     seen_domain,unseen_domain = s.split(' ')
#     run_cmd(seen_domain,unseen_domain)
# for (num_quantizer,per_bit) in [ (2,8),(4,8),(8,6),(8,8),(16,6),(16,8)]:
for (num_quantizer,per_bit) in [ (2,8)]:
    all_bits = num_quantizer*per_bit
    print(all_bits)
    run_cmd('MNIST','USPS',num_quantizer,per_bit,'_'+str(all_bits)+'_bits')
