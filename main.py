import argparse
import os
import math
import datetime
import time

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10
import torchvision.transforms as transforms

from src import builder, eval, utils, sampler


MODEL_NAMES = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


class MultipleTransformation:
    """Modified from MoCo: https://github.com/facebookresearch/moco/blob/main/moco/loader.py#L11-L20"""
    def __init__(self, base_transform, n_augment=2):
        self.base_transform = base_transform
        self.n_augment = n_augment

    def __call__(self, x):
        return torch.stack([self.base_transform(x) for _ in range(self.n_augment)], dim=0)


def adjust_learning_rate(optimizer, epoch, args):
    """Modified from SupCon: https://github.com/HobbitLong/SupContrast/blob/master/util.py#L53-L65"""
    lr = args.lr
    if args.warm:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * ( 1 + math.cos(math.pi * epoch / args.epochs)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    """From SupCon: https://github.com/HobbitLong/SupContrast/blob/master/util.py#L68-L75"""
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train(net, data_loader, train_optimizer, epoch, log, args):
    net.train()
    train_bar = tqdm(data_loader)
    iters_per_epoch = len(data_loader)

    for i, (inputs, labels) in enumerate(train_bar):
        inputs = inputs.cuda(args.gpu, non_blocking=True)
        labels = labels.cuda(args.gpu, non_blocking=True)

        warmup_learning_rate(args, epoch, i, iters_per_epoch, train_optimizer)

        embeddings = net(inputs.view([-1, 3, 32, 32]))
        loss, CL_loss, Sup_loss = builder.loss_func(
            embeddings.view([-1, 2, args.dim]), labels.view([-1, 1]),
            alpha=args.alpha, temperature=args.temperature, args=args
            )

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        train_bar.set_description(
            '[Train:{}/{}] Sup / CL Loss: {:.4f} / {:.4f} ({:.4f})'.format(epoch, args.epochs, Sup_loss.detach().item(), CL_loss.detach().item(), loss.detach().item())
        )
    log.info(
        '[Train:{}/{}] Sup / CL Loss: {:.4f} / {:.4f} ({:.4f})'.format(epoch, args.epochs, Sup_loss.detach().item(), CL_loss.detach().item(), loss.detach().item()),
        print_msg=False
    )




def main(args):
    # create dir
    save_dir = os.path.join(args.save_dir, args.experiment_name)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    # init log
    log = utils.logger(path=save_dir, log_name="log.txt")
    args_table = utils.get_args_table(vars(args))
    log.info(str(args_table)+'\n')

    # gpu and seed
    if args.gpu is not None:
        log.info("Use GPU: {} for training".format(args.gpu))
    torch.backends.cudnn.benchmark = True

    utils.setup_seed(args.seed)
    log.info("Use Seed : {} for replication".format(args.seed))

    ####################
    # model
    log.info("=> Creating model '{}'".format(args.arch))
    model = builder.SupCon(models.__dict__[args.arch], args.dim)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.cuda()

    model = nn.DataParallel(model)

    ####################
    # optimizer
    if args.warm:
        # cosine from SupCon: https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py#L113-L116
        eta_min = args.lr * (args.lr_decay_rate ** 3)
        args.warmup_to = eta_min + (args.lr - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    ####################
    # resume from a checkpoint
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
            best_acc = checkpoint['best_acc']
            log.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        start_epochs = 1
        best_acc = 0

    ####################
    # load data
    log.info('Data preparing')

    if args.data == 'CIFAR100':
        selected_dataset = CIFAR100
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    else:
        selected_dataset = CIFAR10
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    
    # Modified from MoCo: https://github.com/facebookresearch/moco/blob/main/main_moco.py#L328-L338
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean, std)
    ])

    train_data = selected_dataset(root=args.data_dir, train=True, transform=MultipleTransformation(train_transform, args.n_augment))
    test_data = selected_dataset(root=args.data_dir, train=False, transform=test_transform)
    memory_data = selected_dataset(root=args.data_dir, train=True, transform=test_transform)

    args.n_label = len(torch.unique(torch.tensor(memory_data.targets)))
    log.info('Dataset contains {}/{} train/test samples'.format(len(train_data), len(test_data)))
    log.info('# Classes: {}'.format(args.n_label))

    if args.balanced_batch:
        n_batch_instances_within_class = args.batch_size // args.n_label
        log.info('Balanced batch: {} instances within class'.format(n_batch_instances_within_class))
        balanced_batch_sampler = sampler.BalancedBatchSampler(train_data, args.n_label, n_batch_instances_within_class)

        train_loader = DataLoader(train_data, batch_sampler=balanced_batch_sampler, num_workers=args.workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
 
    memory_loader = DataLoader(memory_data, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=True)


    ####################
    # train
    start_time = time.time()
    for epoch in range(start_epochs, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch, args)
        train(model, train_loader, optimizer, epoch, log, args)
        log.info("current lr is {:.5f}".format(optimizer.state_dict()['param_groups'][0]['lr']))

        if epoch % 50 == 0 or epoch==0:
            # test 1 : calculate variance on test dataset
            _ = eval.test_var(model, test_loader, args, normalized=True, log=log)

            # test 2 : linear probing on full dataset 
            test_acc = eval.test_nn(model.module.encoder, memory_loader, test_loader, epoch, args, log)
            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save({'epoch': epoch,'state_dict': model.state_dict(),'optim': optimizer.state_dict(),'best_acc':best_acc,}, os.path.join(save_dir, 'model_best.pt'))

        if epoch % 100 == 0 and epoch > 0:
            torch.save({'epoch': epoch,'state_dict': model.state_dict(),'optim': optimizer.state_dict(),'best_acc':best_acc,}, os.path.join(save_dir, 'model_{:04d}.pt'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log.info('Training time {}\n'.format(total_time_str))
    log.info("Best KNN Test ACC@1  : {:.3f}".format(best_acc))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Cifar100 Self-supervised Training')
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--save_dir', default='checkpoints', type=str, help='path to save checkpoint')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=MODEL_NAMES, help='model architecture: ' + ' | '.join(MODEL_NAMES) + ' (default: resnet18)')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training. ')

    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

    parser.add_argument('--data', default='CIFAR10', type=str, help='CIFAR100 or CIFAR10')
    parser.add_argument('--data_dir', metavar='DIR', help='path to dataset')

    parser.add_argument('--top_k', default=200, type=int, help='Top k most similar images used to predict the label')

    # optimization from SupCon: https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py#L41-L51
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 0.0001)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')

    # warmup from SupCon: https://github.com/HobbitLong/SupContrast/blob/master/main_supcon.py#L111-L112
    parser.add_argument('--warm', action='store_true', help='Use warmup for large-batch training')
    parser.add_argument('--warm_epochs', default=10, type=int, help='Number of warmup epochs for lr')
    parser.add_argument('--warmup_from', default=0.01, type=float, help='Starting warmup')

    # specific configs
    parser.add_argument('--dim', default=128, type=int, help='feature dimension (default : 128)')

    parser.add_argument('--temperature', default=0.1, type=float, help='[p] Softmax temperature')
    parser.add_argument('--alpha', default=0.5, type=float, help='[alpha] Hyper parameter for loss')

    parser.add_argument('--n_label', default=None, type=int, help='[m] The number of classes')
    parser.add_argument('--n_augment', default=2, type=int, help='[p] The number of augmentations')

    parser.add_argument('--batch_size', default=500, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--balanced_batch', default=False, type=bool, help='Use the same number of instances within batch')

    main(parser.parse_args())
