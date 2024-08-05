import sys
import os
import time
from xml.sax.handler import all_features
from cv2 import add
import numpy as np
import pickle
import glob
from datetime import datetime

from datasets import make_dataloader

# pytorch, torch vision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, dataloader
import torchvision.transforms as transforms
from torchvision import models

from icecream import ic

from config import cfg
import argparse
from tqdm import tqdm
from base_models import model_vgg


def extract_feature(data_loader,model,domainname):
	pre_time = time.time()
	cnt = 0
	with torch.no_grad():
		model.eval()
		for n_iter, (img, vid, _, _, idx) in enumerate(tqdm(data_loader)):
			img = img.cuda()

			cnt += img.size(0)
			f = model( img )
			
			if n_iter == 0:
				# features = f.cpu().data.numpy()
				features = f
				lables = vid
				idx_list = idx
			else:
				# features = np.concatenate((features, f.cpu().data.numpy()), axis=0)
				features = torch.cat( (features,f))
				lables = np.concatenate((lables, vid), axis=0)
				idx_list = np.concatenate((idx_list, idx), axis=0)
	ic(cnt)
	features = features.detach().cpu().numpy()

	elapsed = time.time() - pre_time
	print(f"Time for get features:{elapsed//60:.0f}m{elapsed%60:.0f}s\n")

	addr = './fixed_f/vgg/domainnet/'+domainname+'/'
	f_addr = addr+domainname+'_feature.npy'
	cls_addr = addr+domainname+'_label.npy'
	idx_addr = addr+domainname+'_idx.npy'
	
	print('write feature at '+ f_addr)
	os.system('mkdir '+addr)
	np.save(f_addr, features)
	np.save(cls_addr, lables)
	np.save(idx_addr, idx_list)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="ReID Baseline Training")
	parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
	args = parser.parse_args()

	
	model = model_vgg()
	model.eval()
	model.cuda()
	# print(vgg16)
	# exit()


	# cfg.merge_from_file(args.config_file)
	# cfg.merge_from_list(args.opts)
	# cfg.SOLVER.IMS_PER_BATCH = 16
	# cfg.freeze()

	from datasets.domainnet import DomainNet
	from datasets.bases import ImageDataset
	import torchvision.transforms as T
	from datasets.make_dataloader import train_collate_fn

	val_transforms = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


	# for domain in ['painting','quickdraw','real','sketch','infograph','clipart']:
	domain = 'clipart'
	print(domain)

	dataset = DomainNet(root_train="/mnt/hdd1/zhangzhibin/dataset/DomainNet/"+domain+"_train.txt",root_val="/mnt/hdd1/zhangzhibin/dataset/DomainNet/"+domain+"_test.txt")
	train_datset = ImageDataset(dataset.train, val_transforms)
	val_datset = ImageDataset(dataset.test, val_transforms)
	img_loader = DataLoader(
		train_datset, batch_size=64, shuffle=False, num_workers=4,
		collate_fn=train_collate_fn
	)
	extract_feature( img_loader , model , domain+'_train')

	img_loader = DataLoader(
		val_datset, batch_size=64, shuffle=False, num_workers=4,
		collate_fn=train_collate_fn
	)
	extract_feature( img_loader , model , domain+'_val')
