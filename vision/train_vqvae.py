"""
VQ-VAE Training Script
Pre-trains the VQ-VAE on environment screenshots before PPO training.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml
from typing import Optional
import argparse

from vision.vqvae import VQVAE, create_vqvae


class ScreenshotDataset(Dataset):
    """Dataset of environment screenshots for VQ-VAE training."""
    
    def __init__(self, data_dir: str, transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing screenshot images
            transform: Optional transforms to apply
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Find all PNG files
        self.image_files = [
            f for f in os.listdir(data_dir) 
            if f.endswith('.png')
        ]
        
        if len(self.image_files) == 0:
            raise ValueError(f"No PNG files found in {data_dir}")
            
        print(f"Found {len(self.image_files)} images in {data_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Convert to tensor and normalize to [0, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        if self.transform:
            image = self.transform(image)
            
        return image


class VQVAETrainer:
    """Trainer for VQ-VAE model."""
    
    def __init__(
        self,
        model: VQVAE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: str = "runs/vqvae",
        checkpoint_dir: str = "data/checkpoints"
    ):
        """
        Initialize trainer.
        
        Args:
            model: VQ-VAE model
            train_loader: Training data loader
            val_loader: Optional validation data loader
            learning_rate: Learning rate
            device: Device to train on
            log_dir: TensorBoard log directory
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.writer = SummaryWriter(log_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_loss = float('inf')
        self.global_step = 0
        
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, images in enumerate(pbar):
            images = images.to(self.device)
            
            self.optimizer.zero_grad()
            
            loss, loss_dict = self.model.compute_loss(images)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss_dict['total_loss']
            total_recon_loss += loss_dict['recon_loss']
            total_vq_loss += loss_dict['vq_loss']
            
            # Log to TensorBoard
            self.writer.add_scalar('train/batch_loss', loss_dict['total_loss'], self.global_step)
            self.global_step += 1
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'vq': f"{loss_dict['vq_loss']:.4f}"
            })
        
        n_batches = len(self.train_loader)
        return {
            'total_loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches,
            'vq_loss': total_vq_loss / n_batches
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model."""
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        
        for images in self.val_loader:
            images = images.to(self.device)
            loss, loss_dict = self.model.compute_loss(images)
            total_loss += loss_dict['total_loss']
            total_recon_loss += loss_dict['recon_loss']
        
        n_batches = len(self.val_loader)
        return {
            'val_total_loss': total_loss / n_batches,
            'val_recon_loss': total_recon_loss / n_batches
        }
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        path = os.path.join(self.checkpoint_dir, f"vqvae_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "vqvae_best.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model with loss {loss:.4f}")
    
    @torch.no_grad()
    def log_reconstructions(self, epoch: int, num_samples: int = 8):
        """Log reconstruction samples to TensorBoard."""
        self.model.eval()
        
        # Get a batch of images
        images = next(iter(self.train_loader))[:num_samples].to(self.device)
        
        # Get reconstructions
        recon, _, _ = self.model(images)
        
        # Concatenate original and reconstruction
        comparison = torch.cat([images, recon], dim=0)
        
        # Log to TensorBoard
        from torchvision.utils import make_grid
        grid = make_grid(comparison, nrow=num_samples)
        self.writer.add_image('reconstructions', grid, epoch)
    
    def train(self, num_epochs: int, save_interval: int = 10):
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_interval: Save checkpoint every N epochs
        """
        print(f"Training VQ-VAE for {num_epochs} epochs on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step(train_metrics['total_loss'])
            
            # Log epoch metrics
            self.writer.add_scalar('epoch/train_loss', train_metrics['total_loss'], epoch)
            self.writer.add_scalar('epoch/recon_loss', train_metrics['recon_loss'], epoch)
            self.writer.add_scalar('epoch/vq_loss', train_metrics['vq_loss'], epoch)
            
            if val_metrics:
                self.writer.add_scalar('epoch/val_loss', val_metrics['val_total_loss'], epoch)
            
            # Log reconstructions periodically
            if epoch % 10 == 0:
                self.log_reconstructions(epoch)
            
            # Save checkpoints
            is_best = train_metrics['total_loss'] < self.best_loss
            if is_best:
                self.best_loss = train_metrics['total_loss']
                
            if epoch % save_interval == 0 or is_best:
                self.save_checkpoint(epoch, train_metrics['total_loss'], is_best)
            
            print(f"Epoch {epoch}: loss={train_metrics['total_loss']:.4f}, "
                  f"recon={train_metrics['recon_loss']:.4f}, "
                  f"vq={train_metrics['vq_loss']:.4f}")
        
        # Save final model
        self.save_checkpoint(num_epochs, train_metrics['total_loss'])
        self.writer.close()
        
        print(f"\nTraining complete! Best loss: {self.best_loss:.4f}")
        return self.best_loss


def train_from_config(config_path: str = "configs/default.yaml"):
    """Train VQ-VAE using configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vqvae_config = config.get('vqvae', {})
    training_config = config.get('training', {})
    
    # Create model
    model = create_vqvae(config)
    
    # Create dataset
    dataset_dir = training_config.get('dataset_dir', 'data/screenshots')
    dataset = ScreenshotDataset(dataset_dir)
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=vqvae_config.get('batch_size', 64),
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=vqvae_config.get('batch_size', 64),
        shuffle=False,
        num_workers=0
    )
    
    # Create trainer
    trainer = VQVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=vqvae_config.get('learning_rate', 1e-3),
        log_dir=os.path.join(training_config.get('log_dir', 'runs'), 'vqvae'),
        checkpoint_dir=training_config.get('checkpoint_dir', 'data/checkpoints')
    )
    
    # Train
    trainer.train(
        num_epochs=vqvae_config.get('num_epochs', 100),
        save_interval=10
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-VAE model")
    parser.add_argument('--config', '-c', default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--data-dir', '-d', default=None,
                        help='Override dataset directory')
    parser.add_argument('--epochs', '-e', type=int, default=None,
                        help='Override number of epochs')
    
    args = parser.parse_args()
    
    train_from_config(args.config)
