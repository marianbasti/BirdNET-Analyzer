import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import binary_cross_entropy_with_logits
from birdnet_analyzer.torch_model import BirdNetTorchModel
import numpy as np
from typing import Dict, Any, Optional, Callable
from sklearn.metrics import average_precision_score, roc_auc_score
import logging

class AudioDataset(Dataset):
    """Dataset for audio training following Perch data conventions."""
    def __init__(self, audio_data, labels, audio_mask=None):
        self.audio_data = audio_data
        self.labels = labels
        self.audio_mask = audio_mask
    
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        batch = {
            "audio": self.audio_data[idx],
            "label": self.labels[idx]
        }
        if self.audio_mask is not None:
            batch["audio_mask"] = self.audio_mask[idx]
        return batch


class TrainState:
    """Training state following Perch patterns."""
    def __init__(self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer, 
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        self.step = step
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler


class MetricsCollection:
    """Metrics collection following Perch patterns."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.total_loss = 0.0
        self.total_samples = 0
        self.predictions = []
        self.targets = []
        
    def update(self, loss: float, logits: torch.Tensor, labels: torch.Tensor):
        """Update metrics with batch results."""
        batch_size = logits.shape[0]
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        
        # Store predictions and targets for metric computation
        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
            self.predictions.append(probs)
            self.targets.append(labels.cpu().numpy())
    
    def compute(self, prefix: str = "") -> Dict[str, float]:
        """Compute final metrics following Perch conventions."""
        if self.total_samples == 0:
            return {}
            
        avg_loss = self.total_loss / self.total_samples
        
        # Concatenate all predictions and targets
        all_preds = np.concatenate(self.predictions, axis=0)
        all_targets = np.concatenate(self.targets, axis=0)
        
        metrics = {f"{prefix}loss": avg_loss}
        
        try:
            # Compute AUPRC (average precision) following Perch
            if all_targets.ndim == 1:
                # Binary classification
                auprc = average_precision_score(all_targets, all_preds)
                auroc = roc_auc_score(all_targets, all_preds)
            else:
                # Multi-label classification
                auprc = average_precision_score(all_targets, all_preds, average='macro')
                auroc = roc_auc_score(all_targets, all_preds, average='macro')
            
            metrics.update({
                f"{prefix}auprc": auprc,
                f"{prefix}auroc": auroc
            })
            
            # Class-wise metrics (cMAP following Perch)
            if all_targets.ndim > 1:
                class_auprc = []
                class_auroc = []
                for i in range(all_targets.shape[1]):
                    if np.sum(all_targets[:, i]) > 0:  # Only if positive samples exist
                        try:
                            ap = average_precision_score(all_targets[:, i], all_preds[:, i])
                            auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
                            class_auprc.append(ap)
                            class_auroc.append(auc)
                        except:
                            pass
                
                if class_auprc:
                    metrics[f"{prefix}cmap"] = np.mean(class_auprc)
                if class_auroc:
                    metrics[f"{prefix}class_auroc"] = np.mean(class_auroc)
                    
        except Exception as e:
            logging.warning(f"Error computing metrics: {e}")
        
        return metrics

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cuda', 
                loss_fn: Optional[Callable] = None, progress=None, early_stopping_patience=10, 
                log_wandb=False, run_name=None, scheduler_type='ReduceLROnPlateau', 
                resume_from=None, checkpoint_every=5, log_every_steps=10,
                checkpoint_dir="./checkpoints"):
    """
    Train model following Perch training patterns.
    
    Args:
        model: The model to train
        train_loader: Training data loader with batch format {"audio": tensor, "label": tensor}
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Training device
        loss_fn: Loss function (defaults to sigmoid binary cross entropy like Perch)
        progress: Progress tracker
        early_stopping_patience: Patience for early stopping
        log_wandb: Whether to log to wandb
        run_name: Run name for logging
        scheduler_type: Type of learning rate scheduler
        resume_from: Path to checkpoint to resume from
        checkpoint_every: Save checkpoint every N epochs
        log_every_steps: Log metrics every N steps
        checkpoint_dir: Directory to save checkpoints
    """
    import os
    from tqdm import tqdm
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Default to Perch's loss function
    if loss_fn is None:
        loss_fn = F.binary_cross_entropy_with_logits
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Initialize scheduler following Perch patterns
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None
    
    # Initialize training state
    train_state = TrainState(step=0, model=model, optimizer=optimizer, scheduler=scheduler)
    
    # Metrics collections following Perch
    train_metrics = MetricsCollection()
    
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # Wandb logging setup
    if log_wandb:
        import wandb
        wandb.init(project="birdnet-efficientnet", name=run_name)
        wandb.watch(model)
    
    # Resume from checkpoint
    if resume_from and os.path.isfile(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        train_state.step = checkpoint.get('step', start_epoch * len(train_loader))
        logging.info(f"Resumed from checkpoint {resume_from} at epoch {start_epoch}")
    
    # Training loop
    if progress is not None and hasattr(progress, 'tqdm'):
        epoch_iter = progress.tqdm(range(start_epoch, epochs), desc="Training epochs")
    else:
        epoch_iter = range(start_epoch, epochs)
    
    for epoch in epoch_iter:
        # Training phase following Perch patterns
        model.train()
        train_metrics.reset()
        
        step_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            # Extract batch data following Perch conventions
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass with model application like Perch
            optimizer.zero_grad()
            logits = model(audio)
            
            # Compute loss using Perch's default loss function
            loss = loss_fn(logits, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update training state
            train_state.step += 1
            step_count += 1
            
            # Update metrics
            train_metrics.update(loss.item(), logits.detach(), labels.detach())
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item(), "step": train_state.step})
            
            # Log metrics every N steps like Perch
            if step_count % log_every_steps == 0:
                current_metrics = train_metrics.compute(prefix="train_")
                if log_wandb:
                    wandb.log({**current_metrics, "step": train_state.step})
        
        # Compute epoch training metrics
        train_epoch_metrics = train_metrics.compute(prefix="train_")
        
        # Validation phase following Perch evaluation patterns
        val_metrics = evaluate_model(model, val_loader, device, loss_fn, return_metrics=True)
        
        # Learning rate scheduling
        if scheduler_type == 'ReduceLROnPlateau' and scheduler:
            scheduler.step(val_metrics["val_loss"])
        elif scheduler_type == 'CosineAnnealingLR' and scheduler:
            scheduler.step()
        
        # Logging following Perch patterns
        epoch_metrics = {**train_epoch_metrics, **val_metrics, "epoch": epoch + 1}
        
        logging.info(f"Epoch {epoch+1}/{epochs} - " + 
                    " - ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items() 
                               if isinstance(v, (int, float))]))
        
        if log_wandb:
            wandb.log(epoch_metrics)
        
        # Early stopping and best model tracking
        val_loss = val_metrics["val_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save best model
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Checkpoint saving following Perch patterns
        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'epoch': epoch,
                'step': train_state.step,
                'best_val_loss': best_val_loss,
                'metrics': epoch_metrics
            }, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logging.info(f"Training complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
    
    if log_wandb:
        wandb.save(os.path.join(checkpoint_dir, 'best_model.pt'))
        wandb.finish()
    
    return train_state

def evaluate_model(model, loader, device, loss_fn: Optional[Callable] = None, 
                   return_metrics=False) -> Dict[str, float]:
    """
    Evaluate model following Perch evaluation patterns.
    
    Args:
        model: Model to evaluate
        loader: Data loader with batch format {"audio": tensor, "label": tensor}
        device: Evaluation device
        loss_fn: Loss function (defaults to sigmoid binary cross entropy)
        return_metrics: Whether to return detailed metrics
        
    Returns:
        Dictionary of evaluation metrics following Perch conventions
    """
    if loss_fn is None:
        loss_fn = F.binary_cross_entropy_with_logits
        
    model.eval()
    val_metrics = MetricsCollection()
    
    with torch.no_grad():
        for batch in loader:
            # Extract batch data following Perch conventions
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            
            # Model application following Perch patterns (train=False for evaluation)
            logits = model(audio)
            
            # Compute loss
            loss = loss_fn(logits, labels)
            
            # Update metrics
            val_metrics.update(loss.item(), logits, labels)
    
    # Compute final metrics with "val_" prefix following Perch
    metrics = val_metrics.compute(prefix="val_")
    
    if not return_metrics:
        return metrics.get("val_loss", 0.0)
    
    return metrics


def sigmoid_binary_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid binary cross entropy loss function matching Perch's optax version.
    
    This is the default loss function used in Perch training.
    """
    return F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')


def output_head_loss(outputs: torch.Tensor, labels: torch.Tensor, 
                    loss_fn: Callable = sigmoid_binary_cross_entropy,
                    output_head_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
    """
    Compute output head losses following Perch patterns.
    
    Args:
        outputs: Model outputs (logits)
        labels: Ground truth labels  
        loss_fn: Loss function to use
        output_head_weights: Weights for different output heads
        
    Returns:
        Dictionary of losses following Perch conventions
    """
    loss = loss_fn(outputs, labels)
    
    return {
        "loss": loss,
        "output_loss": loss  # Following Perch naming conventions
    }

def focal_loss(logits, targets, gamma=2.0, alpha=0.25, eps=1e-7):
    """Focal loss implementation (kept for backward compatibility)."""
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce_loss
    return loss.mean()


def predict(model, batch, device='cuda'):
    """
    Prediction function following Perch patterns.
    
    Args:
        model: Model for prediction
        batch: Input batch with "audio" key (following Perch conventions)
        device: Device for prediction
        
    Returns:
        Predicted probabilities
    """
    model.eval()
    with torch.no_grad():
        if isinstance(batch, dict):
            audio = batch["audio"].to(device)
        else:
            audio = batch.to(device)  # Backward compatibility
        logits = model(audio)
        return torch.sigmoid(logits)


def create_audio_batch(audio_data, labels, audio_mask=None):
    """
    Create batch following Perch data conventions.
    
    Args:
        audio_data: Audio tensor
        labels: Label tensor
        audio_mask: Optional audio mask
        
    Returns:
        Batch dictionary following Perch format
    """
    batch = {
        "audio": audio_data,
        "label": labels
    }
    if audio_mask is not None:
        batch["audio_mask"] = audio_mask
    return batch
