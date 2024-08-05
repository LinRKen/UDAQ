from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# HashNet(ICCV2017)
# paper [HashNet: Deep Learning to Hash by Continuation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf)
# code [HashNet caffe and pytorch](https://github.com/thuml/HashNet)

def get_config():
    config = {
        "alpha": 0.1,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.001, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[HashNet]",
        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": VGG,
        "dataset": "DomainNet",
        # DomainNet OfficeHome
        "seen_domain": "real",
        "unseen_domain": "clipart",
        # Real_World Clipart 
        # real clipart infograph painting sketch quickdraw
        "epoch": 150,
        "test_map": 5,
        "save_path": "models/HashNet/DomainNet",
        # models/HashNet/DomainNet
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:1"),
        "bit_list": [64],
        'Flag_supervised': True,
    }

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-sd', '--seen_domain')
    parser.add_argument('-ud', '--unseen_domain')
    args = parser.parse_args()
    config['seen_domain'] = args.seen_domain
    config['unseen_domain'] = args.unseen_domain

    config = config_dataset(config)
    return config


class HashNetLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(HashNetLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

        self.scale = 1

    def forward(self, u, y, ind, config):
        u = torch.tanh(self.scale * u)

        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = config["alpha"] * u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

        return loss


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = HashNetLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):
        criterion.scale = (epoch // config["step_continuation"] + 1) ** 0.5

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, scale:%.3f, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"], criterion.scale), end="")

        net.train()

        train_loss = 0
        n_class = config['n_class']
        for image, id_label, ind in train_loader:

            image = image.to(device)
            id_label = id_label.to(device)
            label = torch.nn.functional.one_hot(id_label.view(-1),n_class)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, device=device)
            tst_label = tst_label.view(-1)

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
            trn_label = trn_label.view(-1)

            # print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])

            if mAP > Best_mAP:
                Best_mAP = mAP

                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])
                    print("save in ", config["save_path"])
                    np.save(os.path.join(config["save_path"], config["dataset"] + str(mAP) + "-" + "trn_binary.npy"),
                            trn_binary.numpy())
                    model_id = '_'.join([ config["dataset"],config["seen_domain"],config['unseen_domain'] ])
                    model_path = os.path.join(config["save_path"],  model_id+ "-" + str(mAP) + "-model.pt")
                    print('save at ',model_path)
                    torch.save(net.state_dict(),model_path)
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))
            print(config)



def val(config,bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    model_path = "/mnt/hdd1/zhangzhibin/UDA/CDTrans-master/DeepHash/models/HashNet/DomainNet/DomainNet_real_clipart-0.187993045232194-model.pt"
    checkpoint = torch.load(model_path)
    net.load_state_dict( checkpoint)
    net.eval()


    tst_binary, tst_label = compute_result(test_loader, net, device=device)
    tst_label = tst_label.view(-1)

    # print("calculating dataset binary code.......")\
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
    # trn_binary, trn_label = tst_binary, tst_label
    trn_label = trn_label.view(-1)

    mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),config["topK"])
    print(f'mAP @{config["topK"]:d} = {mAP:.3f}')


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        # train_val(config, bit)
        val(config, bit)
