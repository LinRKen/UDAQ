from audioop import avg
import time
import torch

from loss_func import adaptive_margin_loss,classify_loss

import sys
sys.path.append('..')
from tool import Avg_er



def do_train(epoch,train_loader,model,quantizer,list_optimizer,list_scheduler, label_embeddings, args,logger):
    logger.info('start optimization epoch %d '%(epoch))
    logger.info('lr = %.5f'%( list_optimizer[0].param_groups[-1]['lr']) )
    
    n_train_set = train_loader.dataset.__len__()
    random_noise_stddev = 1e-2
    n_class = args.n_class
    
    batch_size = args.batch_size

    avg_loss = Avg_er('loss')
    avg_loss_f = Avg_er('loss_f')
    avg_loss_cls = Avg_er('loss_cls')
    avg_loss_Q = Avg_er('loss_Q')

    pre = time.time()

    iter_num =  ( n_train_set//batch_size )//2
    n_iter = ( n_train_set+batch_size-1)//batch_size

    for i, data in enumerate(train_loader, 0):
        images, labels = data[0],data[1]
        images = images.cuda()
        labels = labels.cuda()
        
        # images = images + torch.normal( mean = torch.zeros_like(images) , std = random_noise_stddev*torch.ones_like(images) ).cuda()

        for optimizer in list_optimizer:
            optimizer.zero_grad()

        image_embeddings,predicts = model(images)

        loss_f = adaptive_margin_loss( image_embeddings, labels , label_embeddings)
        loss_cls = classify_loss( predicts , labels )
        


        loss = loss_f + 0.1*loss_cls

        if quantizer is not None:
            hard_loss , soft_loss ,joint_loss = quantizer.Q_loss( image_embeddings )
            loss_Q = hard_loss+0.1*soft_loss+0.1*joint_loss
            loss += 0.01*loss_Q

        loss.backward()

        for optimizer in list_optimizer:
            optimizer.step()
        
        N_batch = images.size(0)
        avg_loss.add( loss.item() , N_batch)
        avg_loss_f.add( loss_f.item() , N_batch)
        avg_loss_cls.add( loss_cls.item() , N_batch)

        if quantizer is not None:
            avg_loss_Q.add( loss_Q.item() , N_batch)

        if ( i %iter_num == iter_num-1 ):
            # print(f'epoch {epoch:d} [{i:d}/{n_iter:d}] loss = {avg_loss.mean:.3f} loss_f = {avg_loss_f.mean:.3f} \
            #  loss_cls = {avg_loss_cls.mean:.3f} loss_Q = {avg_loss_Q.mean:.3f}')
            loss_output =  ' '.join( [avg_loss.out_s(),avg_loss_f.out_s(),avg_loss_cls.out_s(),avg_loss_Q.out_s()])
            logger.info(loss_output)
            time_elapsed = time.time() - pre
            logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            pre = time.time()
    for schedule in list_scheduler:
        schedule.step()