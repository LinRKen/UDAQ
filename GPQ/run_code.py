import os
lr = 2e-3
# for s in ['Product Real_World','Real_World Product','Clipart Real_World','Real_World Clipart','Art Real_World','Real_World Art']:
# for s in ['real clipart','real infograph','real painting','real quickdraw','real sketch']:
# for s in ['real clipart']:
# for (n_book,bn_word) in [ (2,8),(4,8),(8,6),(8,8),(16,6),(16,8) ]:
# for (n_book,bn_word) in [ (16,6),(16,8) ]:
for (n_book,bn_word) in [ (16,8) ]:
    # seen_domain,unseen_domain = s.split()
    seen_domain,unseen_domain = ('MNIST','USPS')
    # save_path = '/home/zhangzhibin/data/UDA/CDTrans-master/GPQ/models/OfficeHome/'+seen_domain[0]+'_'+unseen_domain[0]
    # save_path = '/home/zhangzhibin/data/UDA/CDTrans-master/GPQ/models/DomainNet/'
    save_path = '/home/zhangzhibin/data/UDA/CDTrans-master/GPQ/models/MNIST_USPS_20/'
    n_bits = n_book*bn_word
    save_path += '_'.join( [seen_domain[0],unseen_domain[0],str(n_bits)])
    try:
        os.mkdir(save_path)
    except:
        pass
    n_epoch = 50
    cmd = 'python train.py -sd '+seen_domain+' -ud '+unseen_domain
    cmd += ' --save_path '+save_path+'/'
    cmd += ' --dataset MNIST_USPS'
    # cmd += ' --dataset DomainNet'
    cmd += ' --n_book '+str(n_book)
    cmd += ' --bn_word '+str(bn_word)
    cmd += ' --n_epoch '+str(n_epoch)
    cmd += ' --lr '+str(lr)
    # DomainNet OfficeHome MNIST_USPS
    print(cmd)
    os.system(cmd)