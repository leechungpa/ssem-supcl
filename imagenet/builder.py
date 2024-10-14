import torch
import torch.nn as nn

class SupCon(nn.Module):
    def __init__(self, base_encoder, dim=128, T=0.1, alpha=0.5):
        super(SupCon, self).__init__()
        self.T = T
        self.alpha = alpha

        # create the encoder
        self.dim = dim
        self.in_dim = base_encoder().fc.in_features
        self.encoder = base_encoder()
        self.encoder.fc = nn.Identity()

        # build a 2-layer projector
        self.projector = nn.Sequential(nn.Linear(self.in_dim, self.in_dim, bias=False), nn.BatchNorm1d(self.in_dim),
                                    nn.ReLU(inplace=True), nn.Linear(self.in_dim, self.dim))

    def forward(self, x1, x2, labels):

        x = torch.cat([x1, x2], dim=0)
        z = self.projector(self.encoder(x))
        z = nn.functional.normalize(z, dim=1)
                
        batch_size = len(labels)

        # mask for equal instance
        I = torch.eye(batch_size).float().to(labels.device)

        # mask for equal label
        labels = labels.contiguous().view(-1,1)
        mask = torch.eq(labels, labels.T).float().to(labels.device)

        # compute logits
        contrast_count = z.shape[0] // batch_size
        score = torch.mm(z, z.T) / self.T

        # for numerical stability
        logits_max, _ = torch.max(score, dim=1, keepdim=True)
        logits = score - logits_max.detach()

        # tile mask
        mask = mask.repeat(contrast_count, contrast_count)
        I = I.repeat(contrast_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(mask.device), 0)
        mask = mask * logits_mask
        I = I * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        SSL_mask_pos_pairs = I.sum(1)
        SSL_mask_pos_pairs = torch.where(SSL_mask_pos_pairs < 1e-6, 1, SSL_mask_pos_pairs)
        
        Sup_mask_pos_pairs = (mask - I).sum(1)
        Sup_mask_pos_pairs = torch.where(Sup_mask_pos_pairs < 1e-6, 1, Sup_mask_pos_pairs)

        SSL_loss = - (I * log_prob).sum(1) / SSL_mask_pos_pairs
        Sup_loss = - ((mask - I) * log_prob).sum(1) / Sup_mask_pos_pairs

        SSL_loss = SSL_loss.view(contrast_count, batch_size).mean(0)
        Sup_loss = Sup_loss.view(contrast_count, batch_size).mean(0)
        
        loss = self.alpha * SSL_loss + (1. - self.alpha) * Sup_loss
        
        return SSL_loss.mean(), Sup_loss.mean(), loss.mean()

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

