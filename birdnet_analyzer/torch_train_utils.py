import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F
from birdnet_analyzer.torch_model import BirdNetTorchModel

class AudioDataset(Dataset):
    def __init__(self, audio_data, labels):
        self.audio_data = audio_data
        self.labels = labels
    def __len__(self):
        return len(self.audio_data)
    def __getitem__(self, idx):
        return self.audio_data[idx], self.labels[idx]

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda', use_focal_loss=False, gamma=2.0, alpha=0.25, progress=None, early_stopping_patience=10, log_wandb=False, run_name=None, scheduler_type='ReduceLROnPlateau', resume_from=None, checkpoint_every=5, best_model_path='best_model.pt', checkpoint_prefix='checkpoint_finetune_epoch', output_dir=None):
    import os
    from tqdm import tqdm
    import datetime
    # Handle output_dir for all outputs
    if output_dir is not None and output_dir.strip():
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = os.path.join(output_dir, 'best_model.pt')
        checkpoint_prefix = os.path.join(output_dir, 'checkpoint_finetune_epoch')
        log_path = os.path.join(output_dir, 'finetune_log.txt')
        params_path = os.path.join(output_dir, 'finetune_params.csv')
    else:
        log_path = 'finetune_log.txt'
        params_path = 'finetune_params.csv'
    # Logging setup
    def log(msg):
        ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        with open(log_path, "a") as f:
            f.write(f"{ts} {msg}\n")
    # Save params at start
    import csv
    params_dict = {
        "epochs": epochs,
        "learning_rate": lr,
        "device": device
    }
    with open(params_path, "w", newline="") as paramsfile:
        paramswriter = csv.writer(paramsfile)
        paramswriter.writerow(params_dict.keys())
        paramswriter.writerow(params_dict.values())
    best_val_loss = float('inf')
    best_epoch = 0
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler setup
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif scheduler_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
    else:
        scheduler = None
    
    start_epoch = 0
    
    # Weights & Biases logging setup
    if log_wandb:
        import wandb
        if run_name:
            wandb.init(project="birdnet-finetune", name=run_name)
        else:
            wandb.init(project="birdnet-finetune")
        wandb.watch(model)
    
    # Resume from checkpoint support
    if resume_from and os.path.isfile(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from checkpoint {resume_from} at epoch {start_epoch}")
    
    # Progress bar setup
    if progress is not None and hasattr(progress, 'tqdm'):
        epoch_iter = progress.tqdm(range(start_epoch, epochs), desc="Training epochs")
    else:
        epoch_iter = range(start_epoch, epochs)
    
    patience_counter = 0
    
    # Training loop
    for epoch in epoch_iter:
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            # Choose loss function
            if use_focal_loss:
                loss = focal_loss(logits, y, gamma=gamma, alpha=alpha)
            else:
                loss = binary_cross_entropy_with_logits(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            pbar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_loader.dataset)
        val_loss, val_metrics = evaluate_model(model, val_loader, device, use_focal_loss, gamma, alpha, return_metrics=True)
        
        # Learning rate scheduling
        if scheduler_type == 'ReduceLROnPlateau' and scheduler:
            scheduler.step(val_loss)
        elif scheduler and scheduler_type in ['CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'StepLR']:
            scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_metrics['accuracy']:.4f}")
        log(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Logging to Weights & Biases
        if log_wandb:
            log_dict = {
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "val_acc": val_metrics['accuracy'],
                "val_f1": val_metrics['f1'],
                "epoch": epoch + 1
            }
            if not torch.isnan(torch.tensor(val_metrics['auc'])):
                log_dict["val_auc"] = val_metrics['auc']
            if scheduler:
                log_dict["learning_rate"] = optimizer.param_groups[0]['lr']
            wandb.log(log_dict)
        
        # Early stopping and best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, f"{checkpoint_prefix}{epoch+1}.pt")
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1} (best epoch: {best_epoch+1})")
            log(f"Early stopping at epoch {epoch+1} (best epoch: {best_epoch+1})")
            break
    
    print(f"Training complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    log(f"Training complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    if log_wandb:
        wandb.save(best_model_path)
        wandb.finish()

def evaluate_model(model, loader, device, use_focal_loss=False, gamma=2.0, alpha=0.25, return_metrics=False):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if use_focal_loss:
                loss = focal_loss(logits, y, gamma=gamma, alpha=alpha)
            else:
                loss = binary_cross_entropy_with_logits(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    if not return_metrics:
        return avg_loss
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # For multi-label, threshold at 0.5
    pred_labels = (all_preds > 0.5).astype(int)
    acc = accuracy_score(all_targets, pred_labels)
    f1 = f1_score(all_targets, pred_labels, average='macro')
    try:
        auc = roc_auc_score(all_targets, all_preds, average='macro')
    except Exception:
        auc = float('nan')
    metrics = {"accuracy": acc, "f1": f1, "auc": auc}
    return avg_loss, metrics

def focal_loss(logits, targets, gamma=2.0, alpha=0.25, eps=1e-7):
    """
    Focal Loss implementation for handling class imbalance.
    
    Args:
        logits: Raw model outputs (before sigmoid)
        targets: Ground truth binary labels
        gamma: Focusing parameter (higher values focus more on hard examples)
        alpha: Balancing parameter for positive/negative samples
        eps: Small constant for numerical stability
    
    Returns:
        Focal loss value
    """
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce_loss
    return loss.mean()

def predict(model, x, device='cuda'):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        logits = model(x)
        return torch.sigmoid(logits)
