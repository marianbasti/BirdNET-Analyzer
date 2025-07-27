import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math
from einops import rearrange

# --- Mel Spectrogram Preprocessing ---
class BioacousticMelSpecLayer(nn.Module):
    """
    Computes two mel spectrograms tuned for bioacoustics:
    - First: fmin=0, fmax=3000, nfft=2048, hop=288, 96 mel bins (captures 0.3-2s events)
    - Second: fmin=500, fmax=15000, nfft=1024, hop=288, 96 mel bins (captures higher freq events)
    Output: (B, 2, 96, T)
    """
    def __init__(self, sample_rate=48000, spec_shape=(96, 511)):
        super().__init__()
        self.sample_rate = sample_rate
        self.spec_shape = spec_shape
        # Low frequency
        self.mel_low = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=2048,
            hop_length=288,
            f_min=0,
            f_max=3000,
            n_mels=96,
            power=2.0,
            normalized=False,
        )
        # High frequency
        self.mel_high = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=288,
            f_min=500,
            f_max=15000,
            n_mels=96,
            power=2.0,
            normalized=False,
        )
        self.mag_scale = nn.Parameter(torch.tensor(1.23, dtype=torch.float32))

    def forward(self, x):
        # Normalize between -1 and 1
        x = x - x.min(dim=1, keepdim=True)[0]
        x = x / (x.max(dim=1, keepdim=True)[0] + 1e-6)
        x = x - 0.5
        x = x * 2.0
        mel_low = self.mel_low(x)
        mel_high = self.mel_high(x)
        # Ensure both have the same time dimension
        T = min(mel_low.shape[-1], mel_high.shape[-1], self.spec_shape[1])
        mel_low = mel_low[..., :T]
        mel_high = mel_high[..., :T]
        mel_low = mel_low.pow(1.0 / (1.0 + torch.exp(self.mag_scale)))
        mel_high = mel_high.pow(1.0 / (1.0 + torch.exp(self.mag_scale)))
        mel_low = torch.flip(mel_low, dims=[1])
        mel_high = torch.flip(mel_high, dims=[1])
        mel = torch.stack([mel_low, mel_high], dim=1)
        return mel

# --- DeltaNet CAGF-BR Implementation ---
class ShortConvolution(nn.Module):
    def __init__(self, channels, kernel_size, activation="silu"):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels)
        self.activation = activation
    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        # x: (B, L, C)
        x = x.transpose(1,2)  # (B, C, L)
        x = self.conv(x)
        if self.activation == "silu":
            x = F.silu(x)
        elif self.activation == "relu":
            x = F.relu(x)
        elif self.activation == "elu":
            x = F.elu(x)
        x = x.transpose(1,2)  # (B, L, C)
        return x, None

class FusedRMSNormGated(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x, g_vec):
        norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        x = x / (norm + self.eps)
        return x * self.weight + g_vec

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        x = x / (norm + self.eps)
        return x * self.weight

def l2norm(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

class _DepthwiseFIRConv1d(nn.Module):
    def __init__(self, num_heads, head_dim, kernel_size):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.filters = nn.Parameter(torch.randn(num_heads, head_dim, self.kernel_size) * 0.02)
    def forward(self, x):  # (B, L, H, D)
        b, l, h, d = x.shape
        w = rearrange(self.filters, "h d k -> (h d) 1 k")
        x_f = rearrange(x, "b l h d -> b (h d) l")
        x_pad = F.pad(x_f, (self.kernel_size - 1, 0))
        y = F.conv1d(x_pad, weight=w, groups=h * d)
        return rearrange(y, "b (h d) l -> b l h d", h=h)

def _elu_plus_one(x):
    return (F.elu(x, 1.0, False) + 1.0).to(x)

def _sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

def _delta_rule_chunkwise(q, k, v, beta, chunk_size=32):
    b, h, L, d_k = q.shape
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len:
        pad = (0, 0, 0, pad_len)
        q = F.pad(q, pad)
        k = F.pad(k, pad)
        v = F.pad(v, pad)
        beta = F.pad(beta, (0, pad_len))
    L_pad = L + pad_len
    q = l2norm(q)
    k = l2norm(k)
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    q, k, v, k_beta = map(
        lambda t: rearrange(t, "b h (n c) d -> b h n c d", c=chunk_size),
        (q, k, v, k_beta),
    )
    tri = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    tri_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    attn_inv = -(k_beta @ k.transpose(-1, -2)).masked_fill(tri, 0)
    for i in range(1, chunk_size):
        attn_inv[..., i, :i] += (attn_inv[..., i, :, None].clone() * attn_inv[..., :, :i].clone()).sum(-2)
    attn_inv = attn_inv + torch.eye(chunk_size, dtype=attn_inv.dtype, device=q.device)
    u = attn_inv @ v
    w = attn_inv @ k_beta
    S = k.new_zeros(b, h, d_k, v.shape[-1])
    out = torch.zeros_like(v)
    for idx in range(L_pad // chunk_size):
        q_i, k_i = q[:, :, idx], k[:, :, idx]
        attn_local = (q_i @ k_i.transpose(-1, -2)).masked_fill_(tri_strict, 0)
        u_i = u[:, :, idx] - w[:, :, idx] @ S
        out[:, :, idx] = q_i @ S + attn_local @ u_i
        S = S + k_i.transpose(-1, -2) @ u_i
    out = rearrange(out, "b h n c d -> b h (n c) d")
    if pad_len:
        out = out[:, :, :L]
    return out, S

class DeltaNet(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_size=1024,
        num_heads=4,
        expand_k=1.0,
        expand_v=1.0,
        conv_size=4,
        fir_kernel_size_long=64,
        fir_kernel_size_short=5,
        fusion_hidden_mult=2,
        prob_floor=0.02,
        **kwargs,
    ):
        super().__init__()
        self.spec_layer = BioacousticMelSpecLayer()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        # Input dim for projections: 2*96=192 (channels*freqs)
        self.q_proj = nn.Linear(192, self.key_dim, bias=False)
        self.k_proj = nn.Linear(192, self.key_dim, bias=False)
        self.v_proj = nn.Linear(192, self.value_dim, bias=False)
        self.b_proj = nn.Linear(192, num_heads, bias=False)
        self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu")
        self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation="silu")
        self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation="silu")
        self.local_fir_long = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_long)
        self.local_fir_short = _DepthwiseFIRConv1d(num_heads, self.head_v_dim, fir_kernel_size_short)
        self.stat_dim = 16
        gate_input_dim = hidden_size + self.stat_dim
        hidden_gate_dim = hidden_size * fusion_hidden_mult // 2
        self.fusion_gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_gate_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_gate_dim, 4, bias=True),
        )
        self.logit_temperature = nn.Parameter(torch.full((1,), math.log(math.expm1(0.7))))
        self.conv_residual_logit = nn.Parameter(torch.full((num_heads,), -2.0))
        self.g_proj = nn.Linear(192, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=1e-5)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.prob_floor = float(prob_floor)

    @staticmethod
    def _per_head_stats(x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        abs_mean = x.abs().mean(dim=-1, keepdim=True)
        l2 = x.norm(dim=-1, keepdim=True)
        return torch.cat([mean, var, abs_mean, l2], dim=-1)

    def forward(self, x):
        # x: (B, samples)
        mel = self.spec_layer(x)  # (B, 2, 96, T)
        B, C, F, T = mel.shape
        # Ensure both channels have the same time dimension
        # (already handled in BioacousticMelSpecLayer, but let's be explicit)
        # T = mel.shape[-1]
        # Transpose to (B, T, C*F)
        mel_seq = mel.permute(0, 3, 1, 2).reshape(B, T, C * F)  # (B, T, 192)
        # Q/K/V projections + short conv (process sequence)
        q_in = self.q_proj(mel_seq)  # (B, T, key_dim)
        k_in = self.k_proj(mel_seq)
        v_in = self.v_proj(mel_seq)
        # ShortConv expects (B, L, C) -> (B, L, C)
        q_in, _ = self.q_conv1d(q_in)
        k_in, _ = self.k_conv1d(k_in)
        v_in, _ = self.v_conv1d(v_in)
        # Reshape for multi-head: (B, T, num_heads, head_dim)
        q = rearrange(q_in, "b l (h d) -> b l h d", d=self.head_k_dim)
        k = rearrange(k_in, "b l (h d) -> b l h d", d=self.head_k_dim)
        v_direct = rearrange(v_in, "b l (h d) -> b l h d", d=self.head_v_dim)
        beta = self.b_proj(mel_seq).sigmoid()
        beta = torch.clamp(beta, min=1e-6)
        # --- Fix for sequence length mismatch ---
        # All tensors must have shape (B, T, ...)
        seq_len = min(q.shape[1], k.shape[1], v_direct.shape[1], beta.shape[1])
        q = q[:, :seq_len]
        k = k[:, :seq_len]
        v_direct = v_direct[:, :seq_len]
        beta = beta[:, :seq_len]
        # Delta rule expects (B, h, L, d)
        delta_out_t, _ = _delta_rule_chunkwise(
            q=rearrange(q, "b l h d -> b h l d"),
            k=rearrange(k, "b l h d -> b h l d"),
            v=rearrange(v_direct, "b l h d -> b h l d"),
            beta=rearrange(beta, "b l h -> b h l"),
        )
        delta_out = rearrange(delta_out_t, "b h l d -> b l h d")
        local_short = self.local_fir_short(v_direct)
        local_long = self.local_fir_long(v_direct)
        stats_short = self._per_head_stats(local_short)
        stats_long = self._per_head_stats(local_long)
        stats_delta = self._per_head_stats(delta_out)
        stats_value = self._per_head_stats(v_direct)
        stats_vec = torch.cat([stats_short, stats_long, stats_delta, stats_value], dim=-1)
        hs_exp = mel_seq[:, :seq_len].unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        gate_in = torch.cat([hs_exp, stats_vec], dim=-1)
        gate_logits_flat = self.fusion_gate_mlp(rearrange(gate_in, "b l h d -> (b l h) d"))
        # --- Fix: Use torch.nn.functional.softplus, not F.softplus ---
        temperature = torch.nn.functional.softplus(self.logit_temperature) + 1e-4
        gate_logits_flat = gate_logits_flat / temperature
        fusion_logits = rearrange(
            gate_logits_flat,
            "(b l h) c -> b l h c",
            b=gate_in.shape[0],
            l=gate_in.shape[1],
            h=self.num_heads,
        )
        fusion_weights = torch.softmax(fusion_logits, dim=-1)
        if self.prob_floor > 0.0:
            floor_vec = torch.tensor(
                [self.prob_floor, self.prob_floor, 0.0, 0.0],
                dtype=fusion_weights.dtype,
                device=fusion_weights.device,
            )
            fusion_weights = torch.clamp(fusion_weights, min=floor_vec)
            fusion_weights = fusion_weights / fusion_weights.sum(-1, keepdim=True)
        o = (
            fusion_weights[..., 0:1] * local_short
            + fusion_weights[..., 1:2] * local_long
            + fusion_weights[..., 2:3] * delta_out
            + fusion_weights[..., 3:4] * v_direct
        )
        static_gamma = torch.sigmoid(self.conv_residual_logit).to(o.dtype)
        static_gamma = static_gamma[None, None, :, None]
        residual_scale = static_gamma * (1.0 - fusion_weights[..., 0:1])
        o = o + residual_scale * local_short
        g_vec = rearrange(self.g_proj(mel_seq[:, :seq_len]), "b l (h d) -> b l h d", d=self.head_v_dim)
        o = self.o_norm(o, g_vec)
        o = rearrange(o, "b l h d -> b l (h d)")
        o = self.o_proj(o)
        # Pool over time axis (mean pooling)
        pooled = o.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
