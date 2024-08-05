import argparse


class Options:

    def __init__(self):
        # Parse options for processing
        parser = argparse.ArgumentParser(description='options for DPgQ')
        
        parser.add_argument('-path_cp', '--checkpoint_path', default='./checkpoints', type=str)
        parser.add_argument('-data', '--dataset', default='DomainNet', choices=['OfficeHome', 'DomainNet',])
        
        # DomainNet specific arguments
        parser.add_argument('-sd', '--seen_domain', default='real', choices=['real','quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('-ud', '--unseen_domain', default='quickdraw', choices=['real','quickdraw', 'clipart', 'infograph', 'sketch', 'painting'])
        parser.add_argument('--n_class', type=int,default=345)
        
        # Model parameters
        parser.add_argument('-seed', '--seed', type=int, default=0)
        parser.add_argument('-bs', '--batch_size', default=64, type=int)
        parser.add_argument('-nw', '--num_workers', type=int, default=4, help='Number of workers in data loader')
        
        # Optimization parameters
        parser.add_argument('-opt', '--optimizer', type=str, choices=['sgd', 'adam'], default='sgd')
        parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N', help='Number of epochs to train')
        parser.add_argument('-lr', '--lr', type=float, default=0.01, metavar='LR', help='Initial learning rate for optimizer & scheduler')
        parser.add_argument('-mom', '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')

        # quantizer
        parser.add_argument('-K', '--n_codeword', type=int,default=256,  help='The number of codewords pre codebook')
        parser.add_argument('-M', '--num_quantzer', type=int,default=1,  help='The number of quantizer')
        parser.add_argument('-Q_lr','--Q_lr', type=float,default=2e-2,  help='The learning rate of quantizer')
        #

        parser.add_argument('-day','--day', type=str, help='model data')
        parser.add_argument('-id','--id', type=str, help='model ID')
        parser.add_argument('-model_dir','--model_dir', type=str, help='save dir')
        parser.add_argument('-log_dir','--log_dir', type=str, help='save log dir')
        parser.add_argument('-threshold','--threshold', type=float, help='threshold of pseudo label')
        parser.add_argument('-epoch_warmup','--epoch_warmup', type=int, help='epoch warmup of domain adaptation')

        

        self.parser = parser

    
    def parse(self):
        # Parse the arguments
        return self.parser.parse_args()