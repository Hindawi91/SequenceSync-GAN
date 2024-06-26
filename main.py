import os
import argparse
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import torch
import random

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    if config.random_seed != None:
        torch.manual_seed(config.random_seed)
        random.seed(config.random_seed)

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    data_loader_A = None
    data_loader_B = None


    if config.mode == 'train':
        
        data_loader_A = get_loader(image_dir = config.image_dir, image_size = 256, 
               batch_size= int(config.batch_size/2), dataset='Boiling', mode= config.mode, 
               num_workers= config.num_workers, domain = "A")

        data_loader_B = get_loader(image_dir = config.image_dir, image_size = 256, 
                   batch_size= int(config.batch_size/2), dataset='Boiling', mode= config.mode,
                    num_workers= config.num_workers, domain = "B")

        solver = Solver(data_loader_A,data_loader_B, config)

        solver.train()


    elif config.mode == 'test' or config.mode == 'val': #### Take full batch size if testing because you will only use one data loader instead of two

        data_loader_A = get_loader(image_dir = config.image_dir, image_size = 256, 
               batch_size= int(config.batch_size), dataset='Boiling', mode= config.mode, 
               num_workers= config.num_workers, domain = "A")

        data_loader_B = get_loader(image_dir = config.image_dir, image_size = 256, 
                   batch_size= int(config.batch_size), dataset='Boiling', mode= config.mode,
                    num_workers= config.num_workers, domain = "B")

        solver = Solver(data_loader_A,data_loader_B, config)

        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--crop_size', type=int, default=178, help='crop size for the images')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_id', type=float, default=10, help='weight for identity loss')
    parser.add_argument('--lambda_TD', type=float, default=1, help='weight for TD loss')

    
    # Training configuration.
    parser.add_argument('--random_seed', default=None, help='change to any int for producable results')
    
    # parser.add_argument('--batch_size', type=int, default=16, choices=range(2, 999), help='mini-batch size greater than or equal to 2')

    parser.add_argument('--dataset', type=str, default='Boiling')
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')

    parser.add_argument('--td_lr', type=float, default=0.0001, help='learning rate for TD')

    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    # parser.add_argument('--resume_iters', type=int, default=230000, help='resume training from this step') 

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    parser.add_argument('--direction', default='B2A', help='Domain Translation Direction' , choices=['B2A', 'A2B'])

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'val'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--image_dir', type=str, default="../data")
    parser.add_argument('--log_dir', type=str, default='boiling/logs')
    parser.add_argument('--model_save_dir', type=str, default='boiling/models')
    parser.add_argument('--sample_dir', type=str, default='boiling/samples')
    parser.add_argument('--result_dir', type=str, default='boiling/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)
