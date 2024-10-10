import argparse
import argparse
import os

import torch
import torch.nn as nn
import torch.optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10
from torch.utils.data import DataLoader

from src import builder, eval, utils


MODEL_NAMES = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def main(args):
    # create dir
    if os.path.exists(args.save_dir) is not True:
        os.system("mkdir -p {}".format(args.save_dir))

    # init log
    log = utils.logger(path=args.save_dir, log_name=f"log_{args.experiment_name}.txt")
    args_table = utils.get_args_table(vars(args))
    log.info(str(args_table)+'\n')

    # gpu and seed
    if args.gpu is not None:
        log.info("Use GPU: {} for training".format(args.gpu))
    torch.backends.cudnn.benchmark = True

    utils.setup_seed(args.seed)
    log.info("Use Seed : {} for replication".format(args.seed))

    ####################
    # load data
    log.info('Data preparing')

    if args.data == 'CIFAR100':
        selected_dataset = CIFAR100
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
    else:
        selected_dataset = CIFAR10
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean, std)
    ])

    train_data = selected_dataset(root=args.data_dir, train=True, transform=transform)
    test_data = selected_dataset(root=args.data_dir, train=False, transform=transform)

    args.n_label = len(torch.unique(torch.tensor(train_data.targets)))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False) 
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    log.info('Dataset contains {}/{} train/test samples'.format(len(train_data), len(test_data)))

    ####################
    # model
    log.info("=> creating model '{}'".format(args.arch))
    model = builder.SupCon(models.__dict__[args.arch], args.dim)

    model = nn.DataParallel(model)
    
    ####################
    # resume from a checkpoint
    # load from pre-trained, before DistributedDataParallel constructor
    if os.path.isfile(args.pretrained_path):
        log.info("=> loading checkpoint '{}'".format(args.pretrained_path))
        if args.gpu is None:
            checkpoint = torch.load(args.pretrained_path, map_location="cpu")
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.pretrained_path, map_location=loc)
        model.load_state_dict(checkpoint['state_dict'])
        
        log.info("=> loaded pre-trained model '{}'".format(args.pretrained_path))
    else:
        log.info("=> no checkpoint found at '{}'".format(args.pretrained_path))
        return None


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.cuda()

    print('Calculate the variance...')
    total_var, average_within_var, _ = calculate_var(model, train_loader, args.dim, args, normalized=True)
    log.info((f"\n[train] within/between var: {average_within_var:.3f} / {total_var-average_within_var:.3f} ({total_var:.3f})"))
    total_var, average_within_var, _ = calculate_var(model, test_loader, args.dim, args, normalized=True)
    log.info((f"[test] within/between var: {average_within_var:.3f} / {total_var-average_within_var:.3f} ({total_var:.3f})"))



def calculate_var(model, data_loader, dim, args, normalized=False):
    class_mean = torch.zeros(args.n_label, dim, requires_grad=False).cuda(args.gpu, non_blocking=True)
    class_n = torch.zeros(args.n_label, 1, requires_grad=False).cuda(args.gpu, non_blocking=True)

    within_var = torch.zeros(args.n_label, 1, requires_grad=False).cuda(args.gpu, non_blocking=True)
    total_var = torch.zeros(1, requires_grad=False).cuda(args.gpu, non_blocking=True)

    with torch.no_grad():
        # calculate mean vector
        for data in data_loader:
            images, labels = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)
            embeddings = model(images)
            for cnt_class in range(args.n_label):
                class_mean[cnt_class] += embeddings[labels==cnt_class].sum(axis=0)
                class_n[cnt_class] += (labels==cnt_class).sum()

        total_mean = class_mean.sum(axis=0) / class_n.sum()
        class_mean = class_mean / class_n

        # calculate variance
        if normalized:
            total_var = 1 - torch.norm(total_mean, p=2).square()
            within_var = 1 - torch.norm(class_mean, dim=1, p=2).square()
            average_within_var = within_var.mean()
        else:
            for data in data_loader:
                images, labels = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)
                embeddings = model(images)
                total_var += torch.norm(embeddings - total_mean, dim=1, p=2).square().sum()
                for cnt_class in range(args.n_label):
                    within_var[cnt_class] += torch.norm(embeddings[labels==cnt_class] - class_mean[cnt_class], dim=1, p=2).square().sum()
            total_var = total_var / class_n.sum()
            average_within_var = within_var.sum() / class_n.sum()
            within_var = within_var / class_n

    return total_var.detach().item(), average_within_var.detach().item(), within_var.detach()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate transfer performance')
    parser.add_argument('experiment_name', type=str)
    parser.add_argument('--save_dir', default='checkpoints', type=str, help='path to save checkpoint')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=MODEL_NAMES, help='model architecture: ' + ' | '.join(MODEL_NAMES) + ' (default: resnet50)')
    parser.add_argument('--batch_size', '--batch-size', default=512, type=int,metavar='N', help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training. ')

    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 8)')

    # additional configs:
    parser.add_argument('--data_dir', metavar='DIR', help='path to dataset')
    parser.add_argument('--data', default='CIFAR100', type=str, help='Choose from CIFAR100, CIFAR10')
    
    parser.add_argument('--metric', type=str, default='top1', help='top1, class-avg')

    parser.add_argument('--pretrained_path', default='', type=str, help='path to pretrained checkpoint')

    parser.add_argument('--dim', default=128, type=int, help='feature dimension (default : 128)')


    main(parser.parse_args())
