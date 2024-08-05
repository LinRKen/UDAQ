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
    
    loss = -torch.gather( prob_f_cls,1,label.view(-1,1) ).log().mean()
    return loss


def Balance_loss(feat,quantizer):
    n_f = feat.size(0)
    n_quantizer = quantizer.n_quantizer
    n_codeword = quantizer.n_codeword
    
    feat = feat.view(feat.size(0),n_quantizer,-1)
    prob_code = torch.zeros(n_quantizer,n_codeword).cuda()

    for deep in range(n_quantizer):
        dist = -torch.cdist( feat[ :, deep ,: ], quantizer.CodeBooks[deep] )
        prob_code[deep] += torch.nn.functional.softmax( 10*dist , dim = 1 ).sum(dim=0)
    prob_code/= feat.size(0)
    loss = ((prob_code-1/n_codeword)**2).sum(dim=1).sqrt().mean()
    return loss

def Q_gather_loss(T_feat,S_code,quantizer):
    n_quanizer = quantizer.n_quantizer
    n_codeword = quantizer.n_codeword
    n_feat = T_feat.size(0)
    

    T_feat = T_feat.view( n_feat , n_quanizer , -1)
    
    Codebook = quantizer.CodeBooks.view(n_quanizer*n_codeword,-1).data.detach()

    base_M = torch.LongTensor( range(n_quanizer) ).cuda().view(1,-1)
    base_M *= n_codeword

    
    S_code += base_M
    S_code = S_code.view( n_feat , -1 )
    S_Q_feat = Codebook[ S_code , :].view(n_feat,n_quanizer,-1)

    T_feat = T_feat.view(n_feat,n_quanizer,-1)
    return torch.norm(T_feat-S_Q_feat, dim = 2 , p=2).mean()

def trust_loss(T_f,quantizer,prob_CB_cls):
    prob_f_cls = quantizer.predict_cls(T_f,prob_CB_cls)
    
    predict_prob,_ = torch.max( prob_f_cls , dim=1)

    mean = torch.mean( predict_prob )
    # std = torch.std( predict_prob )
    std = 0
    positive_prob = predict_prob[ predict_prob > mean+std ]
    negetive_prob = predict_prob[ predict_prob < mean+std ]
    return -positive_prob.mean()+negetive_prob.mean()
    # print( positive_prob.mean() )
    # return -positive_prob.mean()
    # return negetive_prob.mean()