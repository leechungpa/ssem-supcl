import torch
import torch.nn as nn
import torch.nn.functional as F

class SupCon(nn.Module):
    def __init__(self, base_encoder, dim=128):
        super(SupCon, self).__init__()

        # create the encoder
        self.dim = dim
        self.in_dim = base_encoder().fc.in_features
        self.encoder = base_encoder()
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.encoder.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()

        # build a 2-layer projector
        self.projector = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim, bias=False),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_dim, self.dim)
        )

    def forward(self, x):
        embeddings = self.projector(self.encoder(x))
        return F.normalize(embeddings, p=2, dim=1)

def loss_func(embeddings, labels, alpha, temperature, args):
    """Calculate the convex combination of SupCL and CL losses

    Parameters
    ----------
    embeddings : a tensor of (m*n, p, d) shape
    labels : a tensor of (m*n, 1) shape
    alpha : a value between 0 and 1
    temperature : a value larger then 0
    """
    m_n, p, d = embeddings.shape

    embeddings_flatted = embeddings.view(-1,d)
    mat_exp_products = torch.exp(torch.matmul(embeddings_flatted, embeddings_flatted.T) / temperature)
    mat_softmax = mat_exp_products / mat_exp_products.sum(axis=0)

    pos_mask_cl = torch.kron(torch.eye(m_n), torch.ones(p,p)).cuda(args.gpu, non_blocking=True)
    pos_mask_sup = torch.kron((labels == labels.T).int(), torch.ones(p,p).cuda(args.gpu, non_blocking=True)) - pos_mask_cl

    loss_cl = -torch.log(mat_softmax[pos_mask_cl.bool()]).mean()
    loss_sup = -torch.log(mat_softmax[pos_mask_sup.bool()]).mean()

    return (1-alpha)*loss_sup + alpha*loss_cl, loss_cl, loss_sup

