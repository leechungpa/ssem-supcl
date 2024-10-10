# The original code is from "https://github.com/krafton-ai/mini-batch-cl"

import os
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class Args:
    def __init__(
            self,
            n_for_each_class=[3,3],
            p=2,
            d=5,
            alpha=0.5,
            temperature=1,
            lr_full=0.5, num_steps=1000, logging_step=1000,
            verbos=False, use_mlp=False,
            device="cuda",
            seed=None
            ):
        self.n_for_each_class = n_for_each_class
        self.p = p
        self.d = d
        self.alpha = alpha
        self.temperature = temperature
        self.lr_full = lr_full
        self.num_steps = num_steps
        self.logging_step = logging_step
        self.device = device
        self.use_mlp = use_mlp
        self.verbos = verbos
        self.seed = seed
        
        if verbos:
            print(f"(m,n,p)=({len(n_for_each_class)},{sum(n_for_each_class)},{p})")


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        print("Seeded everything: {}".format(seed))


def generate_gaussian_data(n_for_each_class, p, d):
    """Generate data from the standard Gaussian distribution

    Parameters
    ----------
    n_for_each_class : ex. [5, 3, 2]
    p : the number of augmetations (or views)
    d : instance dimension

    Returns
    -------
    xs : a tensor of (m*n, p, d) shape
    ys : a tensor of (m*n, 1) shape
    """
    xs = []

    for cnt, n in enumerate(n_for_each_class):
        mean = cnt + torch.zeros(d)
        cov = torch.eye(d)
        xs.append(MultivariateNormal(mean, cov).rsample((n, p)))

    xs = torch.concat(xs, dim=0)
    ys = torch.repeat_interleave(torch.arange(len(n_for_each_class)), torch.tensor(n_for_each_class)).view(-1,1)

    # if args.device == 'cuda':
    #     return torch.cuda.FloatTensor(xs), torch.cuda.FloatTensor(ys)
    return xs, ys


def calculate_var(embeddings, ys):
    n_total = embeddings.shape[0] * embeddings.shape[1]

    emb_mean = embeddings.mean(axis=[0,1])

    var_within, var_between = 0, 0

    for cnt in ys.unique():
        emb_within = embeddings[ys.reshape(-1)==cnt]
        emb_within_mean = emb_within.mean(axis=[0,1])

        class_weight = emb_within.shape[0] * emb_within.shape[1] / n_total

        var_within += ((emb_within-emb_within_mean).norm(dim=2)**2).mean() * class_weight
        var_between += (emb_within_mean-emb_mean).norm()**2 * class_weight

    return var_within, var_between



class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.d, args.d, bias=True)
        self.fc2 = nn.Linear(args.d, args.d, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = F.relu(self.fc2(x))
        return x


def loss_func(embeddings, ys, alpha, temperature=1.0):
    """Calculate the convex combination of SupCL and CL losses

    Parameters
    ----------
    embeddings : a tensor of (m*n, p, d) shape
    ys : a tensor of (m*n, 1) shape
    alpha : a value between 0 and 1
    temperature : a value larger then 0
    """
    m_n, p, d = embeddings.shape

    embeddings_flatted = embeddings.view(-1,d)
    mat_exp_products = torch.exp(torch.matmul(embeddings_flatted, embeddings_flatted.T) / temperature)
    mat_softmax = mat_exp_products / mat_exp_products.sum(axis=0)

    pos_mask_cl = torch.kron(torch.eye(m_n), torch.ones(p,p))
    pos_mask_sup = torch.kron((ys == ys.T).int(), torch.ones(p,p)) - pos_mask_cl

    loss_cl = -torch.log(mat_softmax[pos_mask_cl.bool()]).mean()
    loss_sup = -torch.log(mat_softmax[pos_mask_sup.bool()]).mean()

    return (1-alpha)*loss_sup + alpha*loss_cl



def train(**kwargs):
    args = Args(**kwargs)

    set_seed(args.seed)

    xs, ys = generate_gaussian_data(args.n_for_each_class, p=args.p, d=args.d)

    if args.use_mlp:
        model = MLP(args)
        model_params = model.parameters()
    else:
        model = lambda cnt: cnt
        xs.requires_grad_(True)
        model_params = [xs]

    # xs.to(device=args.device)
    # ys.to()
    # if args.device == 'cuda':
    #     model.to('cuda:0')

    optimizer = torch.optim.Adam(model_params, lr=args.lr_full)
    # optimizer = torch.optim.SGD(model_params, lr=args.lr_full)

    for step in tqdm(range(args.num_steps), disable=~args.verbos):
        optimizer.zero_grad()

        embeddings = model(xs)

        embeddings_normalized = F.normalize(embeddings, p=2, dim=2)
        loss = loss_func(embeddings_normalized, ys, alpha=args.alpha, temperature=args.temperature)

        loss.backward()
        optimizer.step()

        if args.verbos:
            if (step % args.logging_step == 0) or (step == args.num_steps-1):
                with torch.no_grad():
                    print({
                        "step":step,
                        "loss":loss_func(embeddings_normalized, ys, alpha=0.5, temperature=1.0).item(),
                        })
                    
    return model(xs.view((-1, args.d))).view(-1, args.p, args.d).detach().cpu(), ys.detach().cpu()


if __name__ == '__main__':
    pass