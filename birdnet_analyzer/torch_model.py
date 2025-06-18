import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class BirdNETMelSpecLayer(nn.Module):
    """
    Computes two mel spectrograms as in BirdNET V2.4 and concatenates them as channels.
    - First: fmin=0, fmax=3000, nfft=2048, hop=278, 96 mel bins
    - Second: fmin=500, fmax=15000, nfft=1024, hop=280, 96 mel bins
    Output: (B, 2, 96, 511)
    """
    def __init__(self, sample_rate=48000, spec_shape=(96, 511), data_format='channels_first'):
        super().__init__()
        self.sample_rate = sample_rate
        self.spec_shape = spec_shape
        self.data_format = data_format
        # Low frequency spectrogram
        self.mel_low = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            win_length=2048,
            hop_length=278,
            f_min=0,
            f_max=3000,
            n_mels=96,
            power=2.0,
            normalized=False,
        )
        # High frequency spectrogram
        self.mel_high = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            win_length=1024,
            hop_length=280,
            f_min=500,
            f_max=15000,
            n_mels=96,
            power=2.0,
            normalized=False,
        )
        # Nonlinear scaling parameter (shared for both)
        self.mag_scale = nn.Parameter(torch.tensor(1.23, dtype=torch.float32))

    def forward(self, x):
        # Normalize between -1 and 1
        x = x - x.min(dim=1, keepdim=True)[0]
        x = x / (x.max(dim=1, keepdim=True)[0] + 1e-6)
        x = x - 0.5
        x = x * 2.0
        # Compute both spectrograms
        mel_low = self.mel_low(x)  # (B, 96, T1)
        mel_high = self.mel_high(x)  # (B, 96, T2)
        # Ensure both have the same time dimension (511)
        mel_low = mel_low[..., :511]
        mel_high = mel_high[..., :511]
        # Nonlinear scaling
        mel_low = mel_low.pow(1.0 / (1.0 + torch.exp(self.mag_scale)))
        mel_high = mel_high.pow(1.0 / (1.0 + torch.exp(self.mag_scale)))
        # Flip frequency axis
        mel_low = torch.flip(mel_low, dims=[1])
        mel_high = torch.flip(mel_high, dims=[1])
        # Stack as channels: (B, 2, 96, 511)
        mel = torch.stack([mel_low, mel_high], dim=1)
        return mel

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        reduced = max(1, int(in_channels * se_ratio))
        self.fc1 = nn.Conv2d(in_channels, reduced, 1)
        self.fc2 = nn.Conv2d(reduced, in_channels, 1)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_res = stride == 1 and in_ch == out_ch
        self.expand = nn.Conv2d(in_ch, mid_ch, 1) if expand_ratio != 1 else nn.Identity()
        self.bn0 = nn.BatchNorm2d(mid_ch)
        self.dwconv = nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, groups=mid_ch)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.se = SqueezeExcite(mid_ch, se_ratio)
        self.project = nn.Conv2d(mid_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        out = self.expand(x)
        out = F.relu6(self.bn0(out))
        out = self.dwconv(out)
        out = F.relu6(self.bn1(out))
        out = self.se(out)
        out = self.project(out)
        out = self.bn2(out)
        if self.use_res:
            out = out + x
        return out

class EfficientNetBackbone(nn.Module):
    def __init__(self, in_ch=2, emb_size=1024):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.blocks = nn.Sequential(
            InvertedResBlock(32, 16, 1, 1),
            InvertedResBlock(16, 24, 2, 6),
            InvertedResBlock(24, 24, 1, 6),
            InvertedResBlock(24, 40, 2, 6),
            InvertedResBlock(40, 40, 1, 6),
            InvertedResBlock(40, 80, 2, 6),
            InvertedResBlock(80, 80, 1, 6),
            InvertedResBlock(80, 112, 1, 6),
            InvertedResBlock(112, 112, 1, 6),
            InvertedResBlock(112, 192, 2, 6),
            InvertedResBlock(192, 192, 1, 6),
            InvertedResBlock(192, 320, 1, 6)
        )
        self.head = nn.Sequential(
            nn.Conv2d(320, emb_size, 1),
            nn.BatchNorm2d(emb_size),
            nn.ReLU6(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x

class BirdNetTorchModel(nn.Module):
    def __init__(self, num_classes, emb_size=1024, spec_shape=(96, 511)):
        super().__init__()
        self.spec_layer = BirdNETMelSpecLayer(spec_shape=spec_shape)
        self.backbone = EfficientNetBackbone(2, emb_size)
        self.classifier = nn.Linear(emb_size, num_classes)
    def forward(self, x):
        try:
            x = self.spec_layer(x)  # (B, 2, 96, 511)
            x = self.backbone(x)
            x = self.classifier(x)
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

# Training and inference utilities would be implemented here as well.
