import torch
import torch.nn.functional as F
from icecream import ic

def classify_loss(predicts, labels):
    return F.cross_entropy(predicts, labels)


def adaptive_margin_loss(image_embeddings, labels, label_embeddings):

    N_img_embed = torch.nn.functional.normalize(image_embeddings)
    N_label_embed = torch.nn.functional.normalize(label_embeddings)

    std_gap = 1 - torch.mm(N_label_embed[labels] , N_label_embed.T )

    dist_map = torch.mm(N_img_embed, N_label_embed.T)

    dist_map -=  torch.sum(N_img_embed * N_label_embed[labels], dim=1 ).view(-1, 1)

    core_loss = torch.clamp(std_gap + dist_map, min=0).mean()

    return core_loss
