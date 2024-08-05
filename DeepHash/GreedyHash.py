from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


# GreedyHash(NIPS2018)
# paper [Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf)
# code [GreedyHash](https://github.com/ssppp/GreedyHash)

def get_config():
    config = {
        "alpha": 0.1,
        "optimizer": {"type": optim.SGD, "epoch_lr_decrease": 100,
                      "optim_params": {"lr": 0.001, "weight_decay": 5e-4, "momentum": 0.9}},

        # "optimizer": {"type": optim.RMSprop, "epoch_lr_decrease": 30,
        #               "optim_params": {"lr": 5e-5, "weight_decay": 5e-4}},

        "info": "[GreedyHash]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": VGG,
        # "net":ResNet,
        # "dataset": "cifar10",
        "epoch": 200,
        "test_map": 5,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [64],
        
        "dataset": "DomainNet",
        # DomainNet OfficeHome
        "seen_domain": "real",
        "unseen_domain": "clipart",
        # Product Real_World
        'save_path': './models/GreedyHash/DomainNet',
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
    config = config_dataset(config)
    if config["dataset"] == "imagenet":
        config["alpha"] = 1
        config["optimizer"]["epoch_lr_decrease"] = 80
    return config


class GreedyHashLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(GreedyHashLoss, self).__init__()
        self.fc = torch.nn.Linear(bit, config["n_class"], bias=False).to(config["device"])
        self.criterion = torch.nn.CrossEntropyLoss().to(config["device"])

    def forward(self, u, onehot_y, ind, config):
        b = GreedyHashLoss.Hash.apply(u)
        # one-hot to label
        y = onehot_y.argmax(axis=1)
        y_pre = self.fc(b)
        loss1 = self.criterion(y_pre, y)
        loss2 = config["alpha"] * (u.abs() - 1).pow(3).abs().mean()
        return loss1 + loss2

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # ctx.save_for_backward(input)
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            # input,  = ctx.saved_tensors
            # grad_output = grad_output.data
            return grad_output


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = GreedyHashLoss(config, bit)

    Best_mAP = 0
    n_class = config['n_class']

    for epoch in range(config["epoch"]):

        lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, lr:%.9f, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, lr, config["dataset"]), end="")

        net.train()

        train_loss = 0
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

    model_path = "/home/zhangzhibin/data/UDA/CDTrans-master/DeepHash/models/GreedyHash/DomainNet/DomainNet_real_clipart-0.3451301272305276-model.pt"
    checkpoint = torch.load(model_path)
    net.load_state_dict( checkpoint)
    net.eval()
    # code_1 = get_code(config,net,train_loader,5,1)
    # code_2 = get_code(config,net,train_loader,5,2)

    # label_1 = torch.LongTensor( [1]*5 ).cpu()
    # label_2 = torch.LongTensor( [2]*5 ).cpu()

    # code_1 = code_1.detach().cpu()
    # code_2 = code_2.detach().cpu()


    # mAP = CalcTopMap(code_1.numpy(), code_2.numpy(), label_1.numpy(), label_2.numpy(),3)

    tst_binary, tst_label = compute_result(test_loader, net, device=device)
    tst_label = tst_label.view(-1)

    # print("calculating dataset binary code.......")\
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
    # trn_binary, trn_label = tst_binary, tst_label
    trn_label = trn_label.view(-1)

    # print("calculating map.......")
    # mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
    #                     config["topK"])
    mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),config["topK"])
    print(f'mAP @{config["topK"]:d} = {mAP:.3f}')


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        # train_val(config, bit)
        val(config,bit)