import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange


    
class IndexPropagationQuantize(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, use_entropy_loss=False,
                 remap=None, unknown_index="random", cosine_similarity=False,
                 entropy_temperature=0.01,
                 sample_minimization_weight=1.0, batch_maximization_weight=1.0):
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.use_entropy_loss = use_entropy_loss
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.cosine_similarity = cosine_similarity
        self.entropy_temperature = entropy_temperature
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # z: [b, d, h, w]
        # embed.weight: [n, d]

        logits = einsum('b d h w, n d -> b n h w', z, self.embedding.weight)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.softmax(logits, dim=1)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros

        dim = 1
        ind = soft_one_hot.max(dim, keepdim=True)[1]
        hard_one_hot = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, ind, 1.0)
        one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot

        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embedding.weight)
        z_q_2 = einsum('b n h w, n d -> b d h w', hard_one_hot, self.embedding.weight)

        quant_loss = torch.mean((z_q - z)**2) + torch.mean((z_q_2.detach()-z)**2) + self.beta * \
                   torch.mean((z_q_2 - z.detach()) ** 2)
        diff = quant_loss

        if self.use_entropy_loss:
            sample_entropy, avg_entropy, entropy_loss= compute_entropy_loss(logits=logits.permute(0, 2, 3, 1).reshape(-1, self.n_e), temperature=self.entropy_temperature, sample_minimization_weight=self.sample_minimization_weight, batch_maximization_weight=self.batch_maximization_weight) # logits [b d h w] -> [b * h * w, n]
            diff = (quant_loss, sample_entropy, avg_entropy, entropy_loss)

        ind = torch.flatten(ind)
        if self.remap is not None:
            ind = ind.reshape(z.shape[0], -1)
            ind = self.remap_to_used(ind)
            ind = ind.reshape(-1, 1)
        return z_q, diff, (None, None, ind)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q