# --- Fix for torchaudio backend on Windows ---
import sys
import platform
if platform.system() == "Windows":
    try:
        import torchaudio
        torchaudio.set_audio_backend("soundfile")
    except Exception as e:
        print("[ERROR] No se pudo establecer el backend 'soundfile' para torchaudio. Instala la librería 'soundfile' con 'pip install soundfile'.")
        raise e

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import random
import os
from torch.utils.data import Dataset, DataLoader
from birdnet_analyzer.torch_model import BirdNetTorchModel, EfficientNetBackbone, BirdNETMelSpecLayer

# 1. Audio Augmentation for Contrastive Learning
class AudioAugment:
    def __init__(self, sample_rate=48000):
        self.sample_rate = sample_rate
    def __call__(self, waveform):
        # Random time shift
        shift = random.randint(0, int(0.1 * self.sample_rate))
        if random.random() > 0.5:
            waveform = torch.roll(waveform, shifts=shift)
        # Add random noise
        if random.random() > 0.5:
            noise = torch.randn_like(waveform) * 0.005
            waveform = waveform + noise
        # Random gain
        if random.random() > 0.5:
            gain = random.uniform(0.8, 1.2)
            waveform = waveform * gain
        # Random frequency/time masking (SpecAugment)
        return waveform

# 2. Dataset for Unlabeled, Variable-Length Audio
class UnlabeledAudioDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=48000, min_len=1.0, max_len=10.0):
        self.audio_paths = []
        self.sample_rate = sample_rate
        self.min_len = min_len
        self.max_len = max_len
        for root, _, files in os.walk(audio_dir):
            for f in files:
                if f.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
                    self.audio_paths.append(os.path.join(root, f))
        self.augment = AudioAugment(sample_rate)
    def __len__(self):
        return len(self.audio_paths)
    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.ndim > 1:
            waveform = waveform[0]  # mono
        # Random crop between min_len and max_len seconds only if audio >= 50s
        total_len = waveform.shape[-1]
        min_samples = int(self.min_len * self.sample_rate)
        max_samples = int(self.max_len * self.sample_rate)
        min_crop_samples = int(50 * self.sample_rate)
        if total_len >= min_crop_samples and total_len > max_samples:
            start = random.randint(0, total_len - max_samples)
            waveform = waveform[start:start+max_samples]
        elif total_len < min_samples:
            pad = min_samples - total_len
            waveform = F.pad(waveform, (0, pad))
        # Two augmentations for contrastive learning
        aug1 = self.augment(waveform.clone())
        aug2 = self.augment(waveform.clone())
        return aug1, aug2

def collate_fn(batch):
    # Pad to max length in batch
    x1, x2 = zip(*batch)
    maxlen = max([a.shape[-1] for a in x1 + x2])
    x1 = [F.pad(a, (0, maxlen - a.shape[-1])) for a in x1]
    x2 = [F.pad(a, (0, maxlen - a.shape[-1])) for a in x2]
    return torch.stack(x1), torch.stack(x2)

# 3. Projection Head for SimCLR
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

# 4. NT-Xent Contrastive Loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        labels = torch.arange(N, device=z1.device)
        labels = torch.cat([labels, labels], dim=0)
        mask = torch.eye(2*N, device=z1.device).bool()
        sim = sim.masked_fill(mask, -9e15)
        positives = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)])
        negatives = sim[~mask].view(2*N, -1)
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        labels = torch.zeros(2*N, dtype=torch.long, device=z1.device)
        loss = F.cross_entropy(logits, labels)
        return loss


# 5. Pretraining Loop
class SimCLRPretrainer:
    def __init__(self, emb_size=1024, proj_dim=128, spec_shape=(96, 511), device='cuda', seed=42, log_wandb=False, run_name=None):
        import numpy as np
        import random
        self.device = device
        self.spec_layer = BirdNETMelSpecLayer(spec_shape=spec_shape).to(device)
        self.backbone = EfficientNetBackbone(2, emb_size).to(device)
        self.proj_head = ProjectionHead(emb_size, proj_dim).to(device)
        self.loss_fn = NTXentLoss()
        self.log_wandb = log_wandb
        self.run_name = run_name
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if self.log_wandb:
            import wandb
            wandb.init(project="birdnet-pretrain", name=run_name, config={"emb_size": emb_size, "proj_dim": proj_dim, "spec_shape": spec_shape, "seed": seed})

    def forward(self, x):
        x = self.spec_layer(x)
        x = self.backbone(x)
        x = self.proj_head(x)
        return x

    def save_for_visualization(self, save_path='pretrained_for_viz.pt'):
        """Save the complete pretraining state including spec layer for visualization."""
        torch.save({
            'spec_layer': self.spec_layer.state_dict(),
            'backbone': self.backbone.state_dict(),
            'proj_head': self.proj_head.state_dict(),
            'config': {
                'emb_size': 1024,  # Assuming default
                'spec_shape': (96, 511),
                'proj_dim': 128
            }
        }, save_path)
        print(f"Complete pretraining state saved to {save_path}")

    def train(self, dataloader, epochs=20, lr=1e-3, save_path='pretrained_backbone.pt', checkpoint_every=5, resume_from=None, use_amp=False, checkpoint_prefix=None):
        import os
        from tqdm import tqdm
        scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
        params = list(self.spec_layer.parameters()) + list(self.backbone.parameters()) + list(self.proj_head.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        start_epoch = 0
        # Resume support
        if resume_from and os.path.isfile(resume_from):
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.spec_layer.load_state_dict(checkpoint['spec_layer'])
            self.backbone.load_state_dict(checkpoint['backbone'])
            self.proj_head.load_state_dict(checkpoint['proj_head'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from checkpoint {resume_from} at epoch {start_epoch}")
        for epoch in range(start_epoch, epochs):
            self.spec_layer.train()
            self.backbone.train()
            self.proj_head.train()
            total_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for x1, x2 in pbar:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                # Check for NaNs or large values
                if torch.isnan(x1).any() or torch.isnan(x2).any():
                    print("[WARNING] NaN detected in input batch. Skipping batch.")
                    continue
                if (x1.abs() > 1e6).any() or (x2.abs() > 1e6).any():
                    print("[WARNING] Large value detected in input batch. Skipping batch.")
                    continue
                optimizer.zero_grad()
                if scaler:
                    with torch.cuda.amp.autocast():
                        z1 = self.forward(x1)
                        z2 = self.forward(x2)
                        loss = self.loss_fn(z1, z2)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    z1 = self.forward(x1)
                    z2 = self.forward(x2)
                    loss = self.loss_fn(z1, z2)
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * x1.size(0)
                pbar.set_postfix({"loss": loss.item()})
            avg_loss = total_loss / len(dataloader.dataset)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            if self.log_wandb:
                import wandb
                wandb.log({"pretrain_loss": avg_loss, "epoch": epoch+1})
            # Save checkpoint
            if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
                ckpt_name = f"{checkpoint_prefix}{epoch+1}.pt" if checkpoint_prefix else f"checkpoint_pretrain_epoch{epoch+1}.pt"
                torch.save({
                    'spec_layer': self.spec_layer.state_dict(),
                    'backbone': self.backbone.state_dict(),
                    'proj_head': self.proj_head.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }, ckpt_name)
        # Save backbone weights
        torch.save(self.backbone.state_dict(), save_path)
        
        # Also save complete state for visualization
        viz_path = save_path.replace('.pt', '_complete.pt')
        self.save_for_visualization(viz_path)
        
        print(f"Pretraining complete. Backbone saved to {save_path}")
        print(f"Complete state for visualization saved to {viz_path}")
        if self.log_wandb:
            import wandb
            wandb.save(save_path)
            wandb.save(viz_path)
