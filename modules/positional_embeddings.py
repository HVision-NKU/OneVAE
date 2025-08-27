import torch
from torch import nn
from transformers.utils import logging

logger = logging.get_logger(__name__)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        # freqs.shape: [max_seq_len, dim//2]
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # emb.shape: [max_seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @property
    def sin_cached(self):
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._sin_cached

    @property
    def cos_cached(self):
        logger.warning_once(
            "The cos_cached attribute will be removed in 4.39. Bear in mind that its contents changed in v4.38. Use "
            "the forward method of RoPE from now on instead. It is not used in the `LlamaAttention` class"
        )
        return self._cos_cached

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # position_ids.shape: [1, seq_len]
        if seq_len is not None:
            logger.warning_once("The `seq_len` argument is deprecated and unused. It will be removed in v4.39.")

        # x: [bs, num_attention_heads, seq_len, head_size]
        # 1, dim//2, 1 -> 1, dim//2, 1
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # 1, seq_len -> 1, 1, seq_len
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # 1, dim//2, seq_len -> 1, seq_len, dim//2
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # 1, seq_len, dim
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids, seq_len=None):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids, seq_len)
        return cos, sin


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim, max_x_position_embeddings, max_y_position_embeddings, base=1000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_x_position_embeddings = max_x_position_embeddings
        self.max_y_position_embeddings = max_y_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_x_seq_len_cached = max_x_position_embeddings
        self.max_y_seq_len_cached = max_y_position_embeddings

        tx = torch.arange(self.max_x_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        tx = tx / self.scaling_factor
        # freqs.shape: [max_x_seq_len, dim//2]
        freqs_x = torch.outer(tx, self.inv_freq)
        # emb_x.shape: [max_x_seq_len, dim]
        emb_x = torch.cat((freqs_x, freqs_x), dim=-1)

        ty = torch.arange(self.max_y_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        ty = ty / self.scaling_factor
        # freqs.shape: [max_x_seq_len, dim//2]
        freqs_y = torch.outer(ty, self.inv_freq)
        # emb_x.shape: [max_x_seq_len, dim]
        emb_y = torch.cat((freqs_y, freqs_y), dim=-1)

        self.register_buffer("_cos_x_cached", emb_x.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_x_cached", emb_x.sin().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_cos_y_cached", emb_y.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_y_cached", emb_y.sin().to(torch.get_default_dtype()), persistent=False)
    
    @property
    def sin_x_cached(self):
        return self._sin_x_cached

    @property
    def cos_x_cached(self):
        return self._cos_x_cached
    @property
    def sin_y_cached(self):
        return self._sin_y_cached

    @property
    def cos_y_cached(self):
        return self._cos_y_cached

    @torch.no_grad()
    def forward(self, x, position_x_ids, position_y_ids, seq_x_len=None, seq_y_len=None):
        # position_x_ids.shape: [1, seq_x_len]

        # x: [bs, num_attention_heads, seq_len, head_size]
        # 1, dim//2, 1 -> 1, dim//2, 1
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_x_ids.shape[0], -1, 1)
        # 1, seq_len -> 1, 1, seq_len
        position_x_ids_expanded = position_x_ids[:, None, :].float()
        position_y_ids_expanded = position_y_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # 1, dim//2, seq_len -> 1, seq_len, dim//2
            freqs_x = (inv_freq_expanded.float() @ position_x_ids_expanded.float()).transpose(1, 2)
            freqs_y = (inv_freq_expanded.float() @ position_y_ids_expanded.float()).transpose(1, 2)
            # 1, seq_len, dim
            emb_x = torch.cat((freqs_x, freqs_x), dim=-1)
            cos_x = emb_x.cos()
            sin_x = emb_x.sin()
            emb_y = torch.cat((freqs_y, freqs_y), dim=-1)
            cos_y = emb_y.cos()
            sin_y = emb_y.sin()
        return cos_x.to(dtype=x.dtype), sin_x.to(dtype=x.dtype), cos_y.to(dtype=x.dtype), sin_y.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    # x: [bs, num_attention_heads, seq_len, head_size]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # cos.shape: 1, seq_len, dim -> 1, 1, seq_len, dim
    # q.shape: bs, num_attention_heads, seq_len, head_size
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids, seq_len=None):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids, seq_len)
        return cos, sin