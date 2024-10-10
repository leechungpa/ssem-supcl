from tqdm import tqdm

import torch
import torch.nn as nn



def test_var(net, data_loader, args, normalized=False, log=None):
    class_mean = torch.zeros(args.n_label, args.dim, requires_grad=False).cuda(args.gpu, non_blocking=True)
    class_n = torch.zeros(args.n_label, 1, requires_grad=False).cuda(args.gpu, non_blocking=True)

    within_var = torch.zeros(args.n_label, 1, requires_grad=False).cuda(args.gpu, non_blocking=True)
    total_var = torch.zeros(1, requires_grad=False).cuda(args.gpu, non_blocking=True)

    with torch.no_grad():
        # calculate mean vector
        for data in data_loader:
            images, labels = data[0].cuda(args.gpu, non_blocking=True), data[1].cuda(args.gpu, non_blocking=True)
            embeddings = net(images)
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
                embeddings = net(images)
                total_var += torch.norm(embeddings - total_mean, dim=1, p=2).square().sum()
                for cnt_class in range(args.n_label):
                    within_var[cnt_class] += torch.norm(embeddings[labels==cnt_class] - class_mean[cnt_class], dim=1, p=2).square().sum()
            total_var = total_var / class_n.sum()
            average_within_var = within_var.sum() / class_n.sum()
            within_var = within_var / class_n

    total_var, average_within_var = total_var.detach().item(), average_within_var.detach().item()
    if log is not None:
        log.info(f"- within/btw var: {average_within_var:.3f} / {total_var-average_within_var:.3f} ({total_var:.3f})")

    return total_var, average_within_var, within_var


def test_nn(net, memory_data_loader, test_data_loader, epoch, args, log=None):
    net.eval()
    total_top1, total_num, feature_bank  = 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _ in tqdm(memory_data_loader, desc='- Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feat = nn.functional.normalize(feature, dim=-1)
            feature_bank.append(feat)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(args.gpu, non_blocking=True), target.cuda(args.gpu, non_blocking=True)
            feature = net(data)
            feat = nn.functional.normalize(feature, dim=-1)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feat, feature_bank)
            # [B, K]
            _, sim_indices = sim_matrix.topk(k=args.top_k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * args.top_k, args.n_label, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, args.n_label), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        
            test_bar.set_description('- KNN Test Epoch: [{}/{}] Acc@1:{:.3f}%'
                                        .format(epoch, args.epochs, total_top1 / total_num * 100))
    if log is not None:
        log.info('- KNN Test Epoch: [{}/{}] Acc@1:{:.3f}%'.format(epoch, args.epochs, total_top1 / total_num * 100), print_msg=False)
    return total_top1 / total_num * 100
