from cv2 import mean
from matplotlib import image
from sympy import ImageSet
import torch as T
from tqdm import tqdm
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from icecream import ic

def squared_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    diff = x.unsqueeze(1) - y.unsqueeze(0)
    return T.sum(diff * diff, -1)

def pqDist_one(C, N_books, g_x, q_x):
    l1, l2 = C.shape
    L_word = int(l2/N_books)
    D_C = T.zeros((l1, N_books), dtype=T.float32)

    q_x_split = T.split(q_x, L_word, 0)
    g_x_split = np.split(g_x.cpu().data.numpy(), N_books, 1)
    C_split = T.split(C, L_word, 1)
    D_C_split = T.split(D_C, 1, 1)

    for j in range(N_books):
        for k in range(l1):
            D_C_split[j][k] =T.norm(q_x_split[j]-C_split[j][k], 2)
            #D_C_split[j][k] = T.norm(q_x_split[j]-C_split[j][k], 2).detach() #for PyTorch version over 1.9
        if j == 0:
            dist = D_C_split[j][g_x_split[j]]
        else:
            dist = T.add(dist, D_C_split[j][g_x_split[j]])
    Dpq = T.squeeze(dist)
    return Dpq

def Indexing(C, N_books, X):
    l1, l2 = C.shape
    L_word = int(l2/N_books)
    x = T.split(X, L_word, 1)
    y = T.split(C, L_word, 1)
    for i in range(N_books):
        diff = squared_distances(x[i], y[i])
        cfg = T.argmin(diff, dim=1)
        min_idx = T.reshape(cfg, [-1, 1])
        if i == 0:
            quant_idx = min_idx
        else:
            quant_idx = T.cat((quant_idx, min_idx), dim=1)
    return quant_idx

def Evaluate_mAP(C, N_books, gallery_codes, query_codes, gallery_labels, query_labels, device, TOP_K=None):
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    with tqdm(total=num_query, desc="Evaluate mAP", bar_format='{desc:<15}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
        for i in range(num_query):
            # Retrieve images from database
            retrieval = (query_labels[i, :] @ gallery_labels.t() > 0).float()

            # Arrange position according to hamming distance
            retrieval = retrieval[T.argsort(pqDist_one(C, N_books, gallery_codes, query_codes[i]))][:TOP_K]

            # Retrieval count
            retrieval_cnt = retrieval.sum().int().item()

            # Can not retrieve images
            if retrieval_cnt == 0:
                continue

            # Generate score for every position
            score = T.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

            # Acquire index
            index = (T.nonzero(retrieval == 1, as_tuple=False).squeeze() + 1.0).float().to(device)

            mean_AP += (score / index).mean()
            pbar.update(1)

        mean_AP = mean_AP / num_query
    return mean_AP

def DoRetrieval(Gallery_loader,Query_loader,device, args, net, C):
    print("Do Retrieval!")

    net.eval()
    with T.no_grad():
        with tqdm(total=len(Gallery_loader), desc="Build Gallery", bar_format='{desc:<15}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, data in enumerate(Gallery_loader, 0):
                gallery_x_batch, gallery_y_batch = data[0].to(device), data[1].to(device)
                outputs = net(gallery_x_batch)
                gallery_c_batch = Indexing(C, args.N_books, outputs[0])
                gallery_y_batch = T.eye(args.num_cls)[gallery_y_batch]
                if i == 0:
                    gallery_c = gallery_c_batch
                    gallery_y = gallery_y_batch
                else:
                    gallery_c = T.cat([gallery_c, gallery_c_batch], 0)
                    gallery_y = T.cat([gallery_y, gallery_y_batch], 0)
                pbar.update(1)

        with tqdm(total=len(Query_loader), desc="Compute Query", bar_format='{desc:<15}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:
            for i, data in enumerate(Query_loader, 0):
                query_x_batch, query_y_batch = data[0].to(device), data[1].to(device)
                outputs = net(query_x_batch)
                query_y_batch = T.eye(args.num_cls)[query_y_batch]
                if i == 0:
                    query_c = outputs[0]
                    query_y = query_y_batch
                else:
                    query_c = T.cat([query_c, outputs[0]], 0)
                    query_y = T.cat([query_y, query_y_batch], 0)
                pbar.update(1)

    mAP = Evaluate_mAP(C, args.N_books, gallery_c.type(T.int), query_c, gallery_y, query_y, device, args.Top_N)
    return mAP

def test_mAP(args,query_set,gallery_set,Q_Source_Flag,G_Source_Flag,net,Q=None,top_k=100,dim=512,logger=None):
    net.eval()

    n_query = query_set.__len__()
    n_gallery = gallery_set.__len__()

    if ( top_k == -1 ):
        top_k = n_gallery

    batch_size = 64

    query_loader = DataLoader(query_set, batch_size=batch_size, shuffle=False,  num_workers=4, pin_memory=True,)
    gallery_loader = DataLoader(gallery_set, batch_size=batch_size, shuffle=False,  num_workers=4, pin_memory=True,)

    total = 0

    pre = time.time()

    with torch.no_grad():
        mean_acc = 0.0
        
        data_database = torch.zeros(n_gallery,dim).cuda()
        if Q is not None:
            data_database_Q_id = torch.LongTensor(n_gallery,Q.n_quantizer).cuda()

        label_database = torch.IntTensor(n_gallery).cuda()
        pos = torch.FloatTensor( range(top_k) ).cuda() + 1

        st = 0
        
        net.Is_source = G_Source_Flag
        for i, data in enumerate(gallery_loader, 0):
            images, labels = data[0],data[1]
            images = images.cuda()
            labels = labels.cuda()

            ed = st + images.size(0)

            image_embeddings, _ = net(images)

            if Q is not None:
                image_embeddings,Q_id = Q(image_embeddings)
                data_database_Q_id[st:ed] = Q_id[:]

            data_database[st:ed] = image_embeddings[:]
            label_database[st:ed] = labels[:]
            st = ed
        
        # if Q is not None:
        #     base_M = torch.LongTensor( range(Q.n_quantizer) ).cuda().view(1,-1)
        #     base_M *= Q.n_codeword
        #     data_database_Q_id += base_M
        #     data_database_Q_id = data_database_Q_id.view(-1)
        cls_acc = torch.zeros(args.n_class).cuda()
        n_query_cls = torch.zeros(args.n_class).cuda()

        net.Is_source = Q_Source_Flag
        for i, data in enumerate(query_loader, 0):
            images, labels = data[0],data[1]
            images = images.cuda()
            labels = labels.cuda()
            
            image_embeddings,_ = net(images)


            
            # if Q is not None:
            #     dist_CB = torch.FloatTensor(images.size(0),Q.n_quantizer,Q.n_codeword).cuda()
            #     part_embeddings = image_embeddings.view(images.size(0),Q.n_quantizer,-1)
            #     for i in range(Q.n_quantizer):
            #         dist_CB[:,i,:] = torch.cdist( part_embeddings[:,i,:] , Q.CodeBooks[i] )
            #     dist_CB = dist_CB.view(images.size(0),-1)



            #     now_dist = dist_CB[ : , data_database_Q_id ]
            #     now_dist = now_dist.view( images.size(0) , Q.n_quantizer,-1)
            #     now_dist = torch.sum(now_dist,dim=1)
            #     # now_dist = torch.gather( dist_CB , batch_database_Q_id ,  dim=1, )
            #     # now_dist = now_dist.view( images.size(0) , -1,Q.n_quantizer)
            #     # now_dist = torch.sum(now_dist,dim=2)
            #     # print(now_dist.size())
            #     # exit()
            # else:
            # now_dist = torch.cdist( image_embeddings , data_database , p=2 )
            now_dist = -torch.mm( image_embeddings , data_database.t() )


            batch_query = images.size(0)

            # ic( now_dist.size())

            new_id = torch.argsort( now_dist.view(batch_query,-1) , dim = 1 )

            # ic(new_id.size())
            # ic(label_database.size())
            now_label = torch.gather( label_database.repeat(batch_query,1) , 1 , new_id  )
            
            cnt_take = ( now_label == labels.view(-1,1) ).float()

            cnt_take = cnt_take[ : , 0:top_k ]
            sum_take = torch.cumsum( cnt_take , 1 )
            all_take = cnt_take.sum(dim=1)
            all_take = torch.clamp( all_take , min = 1e-5 )
            
            now_acc = torch.sum( 
            torch.div(  sum_take , pos )
            *cnt_take , dim=1 
            )/all_take

            # for i in range(labels.size(0)):
            #     cls_acc[labels[i]] += now_acc[i]
            #     n_query_cls[labels[i]] += 1

            now_acc = now_acc.sum().item()
            mean_acc = mean_acc + now_acc
            total += batch_query

            

            # if ( i%query_iter == query_iter-1 ):
            #     print('query finished %d  mean_acc = %.5f'%(total,mean_acc/total))
    # ic(mean_acc)
    mean_acc /= total
    # print(f'map@{top_k:d} = {mean_acc:.3f}')
    logger.info(f'map@{top_k:d} = {mean_acc:.3f}')
    time_elapsed = time.time() - pre
    # print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # cls_acc /= n_query_cls
    # outputs=''
    # for i in range(args.n_class):
    #     outputs += f'{cls_acc[i].item():.3f} '
    # logger.info(outputs)
    # outputs=''
    # for i in range(args.n_class):
    #     outputs += f'{n_query_cls[i].item():.0f} '
    # logger.info(outputs)
    return mean_acc
