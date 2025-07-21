import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math

class BirdNETMelSpecLayer(nn.Module):
    """
    Computes mel spectrogram similar to Whisper preprocessing for BirdNET.
    Creates a single mel spectrogram with 80 mel bins (Whisper standard) instead of dual spectrograms.
    Output: (B, 80, 3000) - matching Whisper's expected input format
    """
    def __init__(self, sample_rate=48000, n_mels=80, n_fft=2048, hop_length=160, data_format='channels_first'):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.data_format = data_format
        
        # Single mel spectrogram following Whisper's preprocessing
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=0,
            f_max=sample_rate // 2,
            n_mels=n_mels,
            power=2.0,
            normalized=False,
        )
        
        # Whisper uses log mel spectrograms
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

    def forward(self, x):
        # Normalize input audio
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-6)
        
        # Compute mel spectrogram
        mel = self.mel_transform(x)  # (B, n_mels, T)
        
        # Convert to dB scale (log mel)
        mel = self.to_db(mel)
        
        # Ensure fixed time dimension (3000 frames for 3 seconds at 48kHz with hop=160)
        target_frames = 3000
        if mel.shape[-1] > target_frames:
            mel = mel[..., :target_frames]
        elif mel.shape[-1] < target_frames:
            pad_width = target_frames - mel.shape[-1]
            mel = F.pad(mel, (0, pad_width))
        
        # Normalize the spectrogram
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        return mel  # (B, 80, 3000)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer similar to Whisper's implementation."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(attn_output)

class TransformerBlock(nn.Module):
    """Transformer block similar to Whisper's encoder block."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attn(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding similar to Whisper."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class WhisperBackbone(nn.Module):
    """
    Whisper-style encoder backbone for audio feature extraction.
    Takes mel spectrograms as input and produces embeddings.
    """
    def __init__(self, n_mels=80, d_model=512, n_heads=8, n_layers=6, d_ff=2048, emb_size=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_mels = n_mels
        
        # Convolutional layers to reduce time dimension (similar to Whisper)
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final projection to embedding size
        self.ln_post = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, emb_size)
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: (B, n_mels, T) = (B, 80, 3000)
        
        # Convolutional layers
        x = F.gelu(self.conv1(x))  # (B, d_model, T)
        x = F.gelu(self.conv2(x))  # (B, d_model, T//2)
        
        # Transpose for transformer: (B, T//2, d_model)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer norm
        x = self.ln_post(x)
        
        # Global average pooling over time dimension
        x = x.transpose(1, 2)  # (B, d_model, T//2)
        x = self.pool(x).squeeze(-1)  # (B, d_model)
        
        # Final projection
        x = self.proj(x)  # (B, emb_size)
        
        return x

class BirdNetTorchModel(nn.Module):
    def __init__(self, num_classes, emb_size=1024, n_mels=80, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.spec_layer = BirdNETMelSpecLayer(n_mels=n_mels)
        self.backbone = WhisperBackbone(
            n_mels=n_mels, 
            d_model=d_model, 
            n_heads=n_heads, 
            n_layers=n_layers, 
            emb_size=emb_size
        )
        self.classifier = nn.Linear(emb_size, num_classes)
        
    def forward(self, x):
        try:
            x = self.spec_layer(x)  # (B, 80, 3000)
            x = self.backbone(x)    # (B, emb_size)
            x = self.classifier(x)  # (B, num_classes)
            return x
        except RuntimeError as e:
            import torch
            if 'out of memory' in str(e).lower():
                print("[ERROR] CUDA out of memory. Emptying cache.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError("CUDA out of memory. Try reducing batch size or input size.") from e
            else:
                raise

