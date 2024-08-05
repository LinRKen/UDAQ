from utils.GPQ_network import *
from utils.Functions import *
from utils import cifar10 as ci10
from utils.RetrievalTest import *
from utils.OfficeHome import deal_OfficeHome
from utils.MNIST_USPS import deal_MNIST_USPS
from utils.DomainNet import deal_DomainNet
import numpy as np

import argparse


def see_pic(img:np.array,addr):
    from PIL import Image
    from utils.DomainNet import anti_color_preprocessing
    img = np.expand_dims(img,0)
    img = anti_color_preprocessing(img)
    img = img[0]
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    img.save(addr)

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--seen_domain')
    parser.add_argument('-ud', '--unseen_domain')
    parser.add_argument('--dataset')
    parser.add_argument('--n_book',type=int)
    parser.add_argument('--bn_word',type=int)
    parser.add_argument('--intn_word',type=int,default=-1)
    args = parser.parse_args()

    # python Demo.py -sd MNIST -ud USPS --dataset MNIST_USPS --n_book 16 --bn_word 8
    # python Demo.py -sd real -ud clipart --dataset DomainNet --n_book 8 --bn_word 8
    # clipart infograph painting quickdraw sketch

    print(args.bn_word)
    bn_word = args.bn_word
    n_book = args.n_book
    args.intn_word = 2**bn_word
    n_bits = bn_word*n_book
    seen_domain = args.seen_domain
    unseen_domain = args.unseen_domain
    print("num_Codewords: 2^%d, num_Codebooks: %d, Bits: %d" % (bn_word, n_book, n_bits))
    # Gallery_x, Query_x = ci10.prepare_data(data_dir, False)

    model_load_path = './models/MNIST_USPS_4/M_U_128/20.ckpt'
    # model_load_path = './models/DomainNet/r_c_64/30.ckpt'
    S_domain = args.seen_domain
    T_domain = args.unseen_domain
    if args.dataset == 'OfficeHome':
        (Source_x, Source_y, Target_x),(Gallery_x, Query_x,label_Similarity) = deal_OfficeHome(S_domain,T_domain)
    elif args.dataset == 'DomainNet':
        (Source_x, Source_y, Target_x),(Gallery_x, Query_x,label_Similarity) = deal_DomainNet(S_domain,T_domain)
    elif args.dataset == 'MNIST_USPS':
        (Source_x, Source_y, Target_x),(Gallery_x, Query_x,label_Similarity) = deal_MNIST_USPS()
    # Product Real_World Clipart Art
    # label_Similarity = label_Similarity.todense()
    # Query_x = Query_x[:120]
    # Gallery_x = Gallery_x[:110]
    # print(n_CLASSES)

    print(Gallery_x.shape)
    print(Query_x.shape)
    # print('label_Similarity = ',label_Similarity.shape)
    top_k = Gallery_x.shape[0]

    # print(n_DB)
    # from PIL import Image
    # dir = "/mnt/hdd1/zhangzhibin/dataset/DomainNet/clipart/airplane/clipart_002_000001.jpg"
    # img = Image.open( dir ).convert('RGB')
    # img = np.array(img)
    # from utils.DomainNet import color_preprocessing
    # img = np.expand_dims(img,0)
    # img = color_preprocessing(img)
    # see_pic(Gallery_x[0],'G_x.png')
    # see_pic(Query_x[0],'Q_x.png')
    # see_pic(img[0],'Q_test.png')
    # exit()


    Net = GPQ(training=training_flag,args=args)
    feature = Net.F(x)
    Z = Net.Z
    saver = tf.train.Saver(tf.global_variables(),max_to_keep=200)

    with tf.Session(config=config) as sess_val:
        saver.restore(sess_val, model_load_path)
        mAP = PQ_retrieval(sess_val, x, training_flag, feature, Z, n_book, Gallery_x, Query_x, label_Similarity, True, TOP_K=top_k)
        print(model_load_path+" mAP: %.3f"%(mAP))

if __name__ == '__main__':
    run()