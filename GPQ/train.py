from email.policy import default
from utils.GPQ_network import *
from utils.Functions import *
from utils import cifar10 as ci10
from utils.RetrievalTest import *
from utils.OfficeHome import deal_OfficeHome,anti_color_preprocessing
from utils.DomainNet import deal_DomainNet
from utils.MNIST_USPS import deal_MNIST_USPS
from icecream import ic

import argparse


# def val(args,Gallery_x,config,model_load_path,n_book,Query_x,label_Similarity):
#     top_k = Gallery_x.shape[0]
#     Net_val = GPQ(training=training_flag,args=args)
#     feature_val = Net_val.F(x)
#     Z_val = Net_val.Z
#     saver = tf.train.Saver(tf.global_variables())

#     with tf.Session(config=config) as sess_val:
#         saver.restore(sess_val, model_load_path)
#         mAP = PQ_retrieval(sess_val, x, training_flag, feature_val, Z_val, n_book, Gallery_x, Query_x, label_Similarity, True, TOP_K=top_k)
#         print(model_load_path+" mAP: %.3f"%(mAP))

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--seen_domain')
    parser.add_argument('-ud', '--unseen_domain')
    parser.add_argument('--save_path')
    parser.add_argument('--dataset')
    parser.add_argument('--n_book',type=int)
    parser.add_argument('--bn_word',type=int)
    parser.add_argument('--intn_word',type=int,default=-1)
    parser.add_argument('--n_epoch',type=int,default=-1)
    parser.add_argument('--lr',type=float)
    args = parser.parse_args()

    bn_word = args.bn_word
    n_book = args.n_book
    args.intn_word = 2**bn_word
    n_bits = bn_word*n_book
    total_epochs = args.n_epoch
    lr = args.lr
    

    print("num_Codewords: 2^%d, num_Codebooks: %d, Bits: %d" % (bn_word, n_book, n_bits))
    
    # Source_x, Source_y, Target_x = ci10.prepare_data(data_dir, True)
    # Gallery_x, Query_x = ci10.prepare_data(data_dir, False)

    # Source_x, Source_y, Target_x = ci10.prepare_data(data_dir, True)
    # Gallery_x, Query_x = ci10.prepare_data(data_dir, False)

    S_domain = args.seen_domain
    T_domain = args.unseen_domain
    if args.dataset == 'OfficeHome':
        (Source_x, Source_y, Target_x),(Gallery_x, Query_x,label_similarity) = deal_OfficeHome(S_domain,T_domain)
    elif args.dataset == 'DomainNet':
        (Source_x, Source_y, Target_x),(Gallery_x, Query_x,label_similarity) = deal_DomainNet(S_domain,T_domain)
    elif args.dataset == 'MNIST_USPS':
        (Source_x, Source_y, Target_x),(Gallery_x, Query_x,label_similarity) = deal_MNIST_USPS()
        

    model_save_path = args.save_path
    print('S_domain = ',S_domain)
    print('T_domain = ',T_domain)
    print('model_save_path = ',model_save_path)

    # for i in range(100):
    #     print(Source_y[i],end=' ')
    # print('')

    print('Source_x.shape = ',Source_x.shape)
    print('Source_y.shape = ',Source_y.shape)

    # Source_x = anti_color_preprocessing( Source_x )
    # data = Source_x[0]
    # from PIL import Image
    # img = Image.fromarray(data.astype('uint8')).convert('RGB')
    # img.save('test_OfficeHome.png')
    # exit()

    print('Target_x.shape = ',Target_x.shape)

    print('Gallery_x.shape = ',Gallery_x.shape)
    print('Query_x.shape = ',Query_x.shape)

    # print(label_similarity.shape)

    Net = GPQ(training=training_flag,args=args)
    Prototypes = Intra_Norm(Net.Prototypes, n_book)
    Z = Soft_Assignment(Prototypes, Net.Z, n_book, alpha)

    feature_S = Net.F(x)
    feature_T = flip_gradient(Net.F(x_T))

    feature_S = Intra_Norm(feature_S, n_book)
    feature_T = Intra_Norm(feature_T, n_book)

    descriptor_S = Soft_Assignment(Z, feature_S, n_book, alpha)

    logits_S = Net.C(feature_S * beta, tf.transpose(Prototypes) * beta)

    hash_loss = N_PQ_loss(labels_Similarity=label_Mat, embeddings_x=feature_S, embeddings_q=descriptor_S, n_book=n_book)
    cls_loss = CLS_loss(label, logits_S)
    entropy_loss = SME_loss(feature_T * beta, tf.transpose(Prototypes) * beta, n_book)

    cost = hash_loss + lam_1*entropy_loss + lam_2*cls_loss

    pretrained_mat = scipy.io.loadmat(ImagNet_pretrained_path)

    var_F = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Fixed_VGG')

    decayed_lr = tf.train.exponential_decay(lr, global_step, 700, 0.95, staircase=True)
    # decayed_lr = tf.train.exponential_decay(0.0001, global_step, 100, 0.95, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=decayed_lr, beta1=0.5).minimize(loss=cost)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate=decayed_lr, use_locking=False,name='GradientDescent').minimize(loss=cost)

    saver = tf.train.Saver(tf.global_variables(),max_to_keep=500)

    with tf.Session(config=config) as sess:


        sess.run(tf.global_variables_initializer())

        print("Load ImageNet2012 pretrained model")
        for i in range(len(var_F) - 2):
            sess.run(var_F[i].assign(np.squeeze(pretrained_mat[var_F[i].name])))

        # if True:
        #     model_load_path = './models/MNIST_USPS_18/M_U_128/50.ckpt'
        #     saver.restore(sess, model_load_path)
        #     print('load from ',model_load_path)

        total_iter = 0

        for epoch in range(1, total_epochs + 1):

            if epoch == 1:

                # label_Similarity = csr_matrix(scipy.io.loadmat(cifar10_label_sim_path)['label_Similarity'])
                # label_Similarity = label_Similarity.todense()

                num_S = np.shape(Source_x)[0]
                num_T = np.shape(Target_x)[0]

                iteration = int(num_S / batch_size)

            for step in range(iteration):

                total_iter += 1

                rnd_idx_S = np.random.choice(num_S, size=batch_size, replace=False)

                batch_Sx = Source_x[rnd_idx_S]
                batch_Sy = Source_y[rnd_idx_S]

                batch_Sy = np.eye(n_CLASSES)[batch_Sy]
                batch_Sy_Mat = np.matmul(batch_Sy, batch_Sy.transpose())
                batch_Sy_Mat /= np.sum(batch_Sy_Mat, axis=1, keepdims=True)
                

                batch_Sx = data_augmentation(batch_Sx)

                rnd_idx_T = np.random.choice(num_T, size=batch_size, replace=False)
                batch_Tx = Target_x[rnd_idx_T]
                batch_Tx = data_augmentation(batch_Tx)

                _, batch_loss, batch_entropy_loss, batch_closs, batch_hash_loss, batch_lr = sess.run(
                    [train_op, cost, entropy_loss, cls_loss, hash_loss, decayed_lr],
                    feed_dict={x: batch_Sx, label: batch_Sy, label_Mat: batch_Sy_Mat, x_T: batch_Tx,
                               training_flag: True, global_step: total_iter-1})

                if (total_iter) % 10 == 0:
                    print("epoch: %d/%d, iter: %d - (Batch) loss: %.4f, hash: %.4f, cls: %.4f, ent: %.4f, lr: %.5f" % (
                            epoch, total_epochs, total_iter, batch_loss, batch_hash_loss, batch_closs,
                            batch_entropy_loss, batch_lr))
            if epoch % save_term == 0:
                save_path = model_save_path+'%d.ckpt'%(epoch)
                print('Model saved at '+save_path)
                saver.save(sess=sess, save_path=save_path)
                # val(args,Gallery_x,config,save_path,n_book,Query_x,label_similarity)

if __name__ == '__main__':
    run()
