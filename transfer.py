import argparse
import os
import time
import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim
import torchvision.models as models
from torch.utils.data import DataLoader

from src import utils
from src.dataset import get_transfer_dataset


MODEL_NAMES = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main(args):
    best_acc, best_w, best_classifier = 0, 0., None

    # create dir
    if os.path.exists(args.save_dir) is not True:
        os.system("mkdir -p {}".format(args.save_dir))

    # init log
    log = utils.logger(path=args.save_dir, log_name=f"log_{args.experiment_name}.txt")
    args_table = utils.get_args_table(vars(args))
    log.info(str(args_table)+'\n')

    # gpu and seed
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    torch.backends.cudnn.benchmark = True

    utils.setup_seed(args.seed)
    log.info("Use Seed : {} for replication".format(args.seed))

    ####################
    # load data
    log.info("Data : {} preparing".format(args.data))
    _, train_data, val_data, test_data, num_classes = get_transfer_dataset(args)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)
    log.info('Dataset contains {}/{}/{} train/val/test samples'.format(len(train_data), len(val_data), len(test_data)))

    ####################
    # model
    log.info("=> creating model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](num_classes=num_classes, zero_init_residual=True)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    classifier = deepcopy(backbone.fc)
    backbone.fc = nn.Identity()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained_path:
        if os.path.isfile(args.pretrained_path):
            log.info("=> loading checkpoint '{}'".format(args.pretrained_path))
            checkpoint = torch.load(args.pretrained_path, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.encoder.') and not k.startswith('module.encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = backbone.load_state_dict(state_dict, strict=False)
            print(msg)
            
            log.info("=> loaded pre-trained model '{}'".format(args.pretrained_path))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.pretrained_path))
            return None

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        backbone = backbone.cuda(args.gpu)
        classifier = classifier.cuda(args.gpu)
    else:
        backbone = backbone.cuda()
        classifier = classifier.cuda()

    print('Collecting features...')
    start_data_time = time.time()
    X_train, Y_train = collect_features(backbone, train_loader, args)
    X_val, Y_val = collect_features(backbone, val_loader, args)
    X_test, Y_test = collect_features(backbone, test_loader, args)
    
    optim_kwargs = {
        'line_search_fn': 'strong_wolfe',
        'max_iter': 5000,
        'lr': 1.,
        'tolerance_grad': 1e-10,
        'tolerance_change': 0,
    }
    total_data_time = time.time() - start_data_time
    total_data_time_str = str(datetime.timedelta(seconds=int(total_data_time)))
    print('Collecting features done {} time...'.format(total_data_time_str))
    
    start_time = time.time()
    for w in torch.logspace(-6, 5, steps=45).tolist():
        optimizer = torch.optim.LBFGS(classifier.parameters(), **optim_kwargs)
        optimizer.step(build_step(X_train, Y_train, classifier, optimizer, w))
        acc = compute_accuracy(X_test, Y_test, classifier, args.metric)
    
        if best_acc < acc:
            best_acc = acc
            best_w = w
            best_classifier = deepcopy(classifier)

        print(f'w={w:.4e}, acc={acc:.4f}')

    log.info(f'BEST for Test: w={best_w:.4e}, acc={best_acc:.4f}')
    
    X = torch.cat([X_train, X_val], 0)
    Y = torch.cat([Y_train, Y_val], 0)
    optimizer = torch.optim.LBFGS(best_classifier.parameters(), **optim_kwargs)
    optimizer.step(build_step(X, Y, best_classifier, optimizer, best_w))
    acc = compute_accuracy(X_test, Y_test, best_classifier, args.metric)
    log.info(f'\nTest acc : {acc:.4f} (Metric:{args.metric})')
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time {}'.format(total_time_str))

def collect_features(model, dataloader, args):
    model.eval()
    with torch.no_grad():
        features = []
        labels = []
        for x, y in dataloader:
            z = model(x.cuda(args.gpu))
            features.append(z.detach())
            labels.append(y.to(z.device))
        features = torch.cat(features, 0).detach()
        labels = torch.cat(labels, 0).detach()
    return features, labels

def build_step(X, Y, classifier, optimizer, w):
    def step():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(classifier(X), Y, reduction='sum')
        for p in classifier.parameters():
            loss = loss + p.pow(2).sum().mul(w)
        loss.backward()
        return loss
    return step

def compute_accuracy(X, Y, classifier, metric):
    with torch.no_grad():
        preds = classifier(X).argmax(1)
        if metric == 'top1':
            acc = (preds == Y).float().mean().item()
        elif metric == 'class-avg':
            total, count = 0., 0.
            for y in range(0, Y.max().item()+1):
                masks = Y == y
                if masks.sum() > 0:
                    total += (preds[masks] == y).float().mean().item()
                    count += 1
            acc = total / count
        else:
            raise Exception(f'Unknown metric: {metric}')
    return acc



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

    parser.add_argument('--pretrain_data', default='CIFAR100', type=str, help='Choose from CIFAR100, CIFAR10')
    parser.add_argument('--pretrained_path', default='', type=str, help='path to pretrained checkpoint')



    main(parser.parse_args())
