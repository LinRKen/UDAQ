import torch

def CB_predict_loss(feat,label,quantizer,prob_CB_cls):
    n_f = feat.size(0)
    n_quantizer = quantizer.n_quantizer
    n_cls = prob_CB_cls.size(-1)
    prob_f_cls = torch.zeros( n_f , n_cls).cuda()
    
    feat = feat.view(feat.size(0),n_quantizer,-1)

    for deep in range(n_quantizer):
        dist = -torch.cdist( feat[ :, deep ,: ], quantizer.CodeBooks[deep] )
        prob_f_code = torch.nn.functional.softmax( 10*dist , dim = 1 )
        prob_f_cls += torch.mm( prob_f_code , prob_CB_cls[deep] )/n_quantizer

    # prob_f_cls = prob_f_cls[:1]
    # label = label[:1]
    # prob_f_cls = prob_f_cls.log()
    # print(prob_f_cls)
    # print(label)
    # print(prob_f_cls[0][label[0]])
    # std = torch.nn.functional.cross_entropy(prob_f_cls.log(), label)
    
    loss = -torch.gather( prob_f_cls,1,label.view(-1,1) ).log().mean()
    # print(std,mine)
    # loss = loss.mean()
    # exit()
    return loss