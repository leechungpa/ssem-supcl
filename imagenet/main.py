import argparse
import os
import math
from tqdm import tqdm
import builtins
from prettytable import PrettyTable
import datetime
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import torch.distributed as dist

from datasets.imagenet import ImageNet
from builder import *
from utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Imagenet100 Supervised Contrastive Training')
parser.add_argument('experiment', type=str)
parser.add_argument('--save_dir', default='checkpoints', type=str, help='path to save checkpoint')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 16)')
parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--lr', '--learning-rate', default=0.3, type=float, metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 0.0001)', dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=2023, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dir', default='', type=str, metavar='PATH', help='Path to dataset')
parser.add_argument('--subset', default=100, type=int, help='Imagenet subset 100 or None')

# specific configs
parser.add_argument('--dim', default=128, type=int, help='feature dimension (default : 128)')
parser.add_argument('--T', default=0.1, type=float, help='softmax temperature (default : 0.1)')

# additional configs
parser.add_argument('--alpha', default=0.5, type=float, help='Hyper parameter for loss')

log = None

def main():
    args = parser.parse_args()
    if args.seed is not None:
        setup_seed(args.seed)
        print("Use Seed : {} for replication".format(args.seed))

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global log
    args.gpu = gpu
        
    if not (args.multiprocessing_distributed and args.gpu != 0):
        # create dir
        save_dir = os.path.join(args.save_dir, args.experiment)
        if os.path.exists(save_dir) is not True:
            os.system("mkdir -p {}".format(save_dir))
        
        # init log
        log = logger(path=save_dir, log_name="log.txt")
        args_table = get_args_table(vars(args))
        log.info(str(args_table)+'\n')

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # create model
    if log is not None:
        log.info("=> Creating model Supcon (alpha : {}) with backbone '{}'".format(args.alpha, args.arch))
        
    model = SupCon(models.__dict__[args.arch], args.dim, args.T, args.alpha)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    # Data parallel or gpu setting
    if args.distributed:
         # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define optimizer  
    # if args.batch_size > 256:
    #     args.warm = True
    # if args.warm:
    #     args.warmup_from = 0.01
    #     args.warm_epochs = 10
    #     eta_min = init_lr * (args.lr_decay_rate ** 3)
    #     args.warmup_to = eta_min + (init_lr - eta_min) * (1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
    optimizer = torch.optim.SGD(model.parameters(), init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

        
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            start_epochs = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim'])
            if log is not None:
                log.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            if log is not None:
                log.info("=> no checkpoint found at '{}'".format(args.resume))
    else:
        start_epochs = 1
    torch.cuda.empty_cache()
    cudnn.benckmark=True

    # data prepare
    if log is not None:
        log.info('Data preparing')
    
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.2, 1.)), 
                                            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = ImageNet(root = args.dir, subset = args.subset, split='train', transform = TwoCropsTransform(train_transform))
    c = len(torch.unique(torch.tensor(train_data.targets)))
    if log is not None:
        log.info('Dataset contains {} train samples'.format(len(train_data)))
        log.info('# Classes: {}'.format(c) + '\n')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
 
    # training
    start_time = time.time()
    for epoch in range(start_epochs, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        
        if log is not None: log.info("current lr is {:.5f}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        
        # train for one epoch
        train(model, train_loader, optimizer, epoch, log, args)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if epoch % 50 == 0:
                save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),'optim': optimizer.state_dict(),}, 
                                filename=os.path.join(save_dir, 'model_{:04d}.pt'.format(epoch)))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if log is not None:
        log.info('Training time {}'.format(total_time_str))
    
def train(net, data_loader, train_optimizer, epoch, log, args):
    net.train()
    total_loss, total_SSL_loss, total_Sup_loss, total_num = 0.0, 0.0, 0.0, 0
    if log is not None: train_bar = tqdm(data_loader)
    else: train_bar = data_loader
    torch.cuda.empty_cache()
    iters_per_epoch = len(data_loader)
    for i, (data, labels) in enumerate(train_bar):
        pos_1, pos_2 = data
        pos_1, pos_2 = pos_1.cuda(args.gpu, non_blocking=True), pos_2.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)
    
        # warmup_learning_rate(args, epoch, i, iters_per_epoch, train_optimizer)
    
        # loss
        SSL_loss, Sup_loss, loss = net(pos_1, pos_2, labels)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += args.batch_size
        total_loss += loss.detach().item() * args.batch_size
        total_SSL_loss += SSL_loss.detach().item() * args.batch_size
        total_Sup_loss += Sup_loss.detach().item() * args.batch_size

        if log is not None: train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} SSL Loss: {:.4f} Sup Loss: {:.4f}'
                                .format(epoch, args.epochs, total_loss / total_num, total_SSL_loss / total_num, total_Sup_loss / total_num))

    if log is not None:
        log.info('Train Epoch : [{}/{}] Avg Loss: {:.4f} SSL Loss: {:.4f} Sup Loss: {:.4f}'
                .format(epoch, args.epochs, total_loss / total_num, total_SSL_loss / total_num, total_Sup_loss / total_num)) 

def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Modified from SupCon: https://github.com/HobbitLong/SupContrast/blob/master/util.py#L53-L65"""
    lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - 1) / args.epochs ))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
#     """From SupCon: https://github.com/HobbitLong/SupContrast/blob/master/util.py#L68-L75"""
#     if args.warm and epoch <= args.warm_epochs:
#         p = (batch_id + (epoch - 1) * total_batches) / \
#             (args.warm_epochs * total_batches)
#         lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr

def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)
    
if __name__ == '__main__':
    main()


