import os
# for s in ['real clipart','real infograph','real painting','real sketch','real quickdraw']:
for s in ['Product Real_World','Real_World Product','Clipart Real_World','Real_World Clipart','Art Real_World','Real_World Art']:
# for s in ['Product Real_World']:
# for s in ['real clipart',]:
    seen_domain,unseen_domain = s.split(' ')
    # cmd = 'CUDA_VISIBLE_DEVICES=0 python Unsupervised_BiHalf.py -sd '+seen_domain+' -ud '+unseen_domain
    cmd = 'python CSQ.py -sd '+seen_domain+' -ud '+unseen_domain
    # cmd = 'python GreedyHash.py -sd '+seen_domain+' -ud '+unseen_domain
    print(cmd)
    os.system(cmd)