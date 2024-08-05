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
    loss = ((prob_code-1/n_codeword)**2).mean()
    # print(prob_code)
    # print(loss)
    return loss