"""
VQ-VAE (Vector Quantized Variational Autoencoder) for Discrete Visual Representations

Based on paper 2312.01203v3: "Harnessing Discrete Representations for Continual RL"

Key features:
- Encoder: CNN that compresses 64x64 RGB images to 6x6 latent grid
- Vector Quantization: Maps continuous latents to discrete codebook entries
- Multi-One-Hot: Converts discrete codes to sparse binary vectors for PPO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer with codebook.
    
    Maps continuous latent vectors to nearest codebook entries,
    producing discrete representations.
    """
    
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 1.0
    ):
        """
        Initialize Vector Quantizer.
        
        Args:
            num_embeddings: Size of the codebook (K in paper)
            embedding_dim: Dimension of each embedding vector
            commitment_cost: β coefficient for commitment loss
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        
        # Initialize embeddings uniformly
        self.embeddings.weight.data.uniform_(
            -1.0 / num_embeddings, 
            1.0 / num_embeddings
        )
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous latent vectors.
        
        Args:
            z: Continuous latent tensor (B, D, H, W)
            
        Returns:
            z_q: Quantized tensor (B, D, H, W)
            indices: Codebook indices (B, H, W)
            vq_loss: VQ loss + commitment loss
        """
        # z: (B, D, H, W) -> (B, H, W, D)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)
        
        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        d = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embeddings.weight.t())
        )
        
        # Find nearest codebook entries
        indices = torch.argmin(d, dim=1)
        
        # Quantize
        z_q = self.embeddings(indices)
        z_q = z_q.view(z.shape)
        
        # Compute VQ loss
        # VQ loss: ||sg[z] - e||^2 (move embeddings toward encoder outputs)
        # Commitment loss: ||z - sg[e]||^2 (encourage encoder to commit)
        vq_loss = F.mse_loss(z_q.detach(), z)  # VQ loss
        commitment_loss = F.mse_loss(z_q, z.detach())  # Commitment loss
        
        loss = vq_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q - z).detach()
        
        # Back to (B, D, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # Reshape indices to (B, H, W)
        indices = indices.view(z.shape[0], z.shape[1], z.shape[2])
        
        return z_q, indices, loss
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Get codebook entries for given indices."""
        z_q = self.embeddings(indices)
        return z_q.permute(0, 3, 1, 2).contiguous()


class Encoder(nn.Module):
    """
    CNN Encoder: 64x64x3 → 6x6x64 latent grid.
    
    Downsamples the input image through convolutional layers
    to produce a compact latent representation.
    
    Architecture: 64 -> 32 -> 16 -> 8 -> 6 (using adaptive pooling)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: list = [32, 64, 128],
        embedding_dim: int = 64,
        latent_size: int = 6
    ):
        super().__init__()
        
        self.latent_size = latent_size
        
        # Convolutional layers: 64 -> 32 -> 16 -> 8
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True)
        )  # 64 -> 32
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(inplace=True)
        )  # 32 -> 16
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels[2]),
            nn.ReLU(inplace=True)
        )  # 16 -> 8
        
        # Adaptive pooling to get exact 6x6
        self.adaptive_pool = nn.AdaptiveAvgPool2d((latent_size, latent_size))
        
        # Final projection to embedding dimension
        self.proj = nn.Conv2d(hidden_channels[2], embedding_dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input image.
        
        Args:
            x: Input image (B, 3, 64, 64)
            
        Returns:
            z: Latent representation (B, embedding_dim, 6, 6)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.proj(x)
        return x


class Decoder(nn.Module):
    """
    CNN Decoder: 6x6x64 → 64x64x3 reconstruction.
    
    Upsamples the latent representation back to image space.
    
    Architecture: 6 -> 8 -> 16 -> 32 -> 64 (using interpolation + conv)
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_channels: list = [128, 64, 32],
        out_channels: int = 3
    ):
        super().__init__()
        
        # Initial projection
        self.proj = nn.Conv2d(embedding_dim, hidden_channels[0], kernel_size=1)
        
        # Upsample layers: 6 -> 8 -> 16 -> 32 -> 64
        self.up1 = nn.Sequential(
            nn.Upsample(size=8, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels[0], hidden_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[0]),
            nn.ReLU(inplace=True)
        )  # 6 -> 8
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[1]),
            nn.ReLU(inplace=True)
        )  # 8 -> 16
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[2]),
            nn.ReLU(inplace=True)
        )  # 16 -> 32
        
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels[2], hidden_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels[2]),
            nn.ReLU(inplace=True)
        )  # 32 -> 64
        
        # Final projection to RGB
        self.final = nn.Sequential(
            nn.Conv2d(hidden_channels[2], out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latent.
        
        Args:
            z_q: Quantized latent (B, embedding_dim, 6, 6)
            
        Returns:
            x_recon: Reconstructed image (B, 3, 64, 64)
        """
        x = self.proj(z_q)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        return x


class VQVAE(nn.Module):
    """
    Complete VQ-VAE model for discrete visual representations.
    
    Architecture:
    - Encoder: 64x64 RGB → 6x6 latent grid
    - Vector Quantizer: Continuous → Discrete codes
    - Decoder: 6x6 latent → 64x64 reconstruction
    - Multi-One-Hot: Discrete codes → Binary sparse vector
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        commitment_cost: float = 1.0,
        latent_grid_size: int = 6
    ):
        """
        Initialize VQ-VAE.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            embedding_dim: Dimension of latent embeddings
            num_embeddings: Codebook size
            commitment_cost: β for commitment loss
            latent_grid_size: Size of latent grid (6x6)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.latent_grid_size = latent_grid_size
        
        # Multi-one-hot output dimension: grid_size^2 * codebook_size
        self.multi_one_hot_dim = latent_grid_size * latent_grid_size * num_embeddings
        
        self.encoder = Encoder(in_channels, embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VQ-VAE.
        
        Args:
            x: Input image (B, 3, 64, 64), values in [0, 1]
            
        Returns:
            x_recon: Reconstructed image (B, 3, 64, 64)
            indices: Codebook indices (B, 6, 6)
            vq_loss: VQ + commitment loss
        """
        # Encode
        z = self.encoder(x)
        
        # Quantize
        z_q, indices, vq_loss = self.quantizer(z)
        
        # Decode
        x_recon = self.decoder(z_q)
        
        return x_recon, indices, vq_loss
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to quantized latent.
        
        Args:
            x: Input image (B, 3, 64, 64)
            
        Returns:
            z_q: Quantized latent (B, embedding_dim, 6, 6)
            indices: Codebook indices (B, 6, 6)
        """
        z = self.encoder(x)
        z_q, indices, _ = self.quantizer(z)
        return z_q, indices
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latent to image.
        
        Args:
            z_q: Quantized latent (B, embedding_dim, 6, 6)
            
        Returns:
            x_recon: Reconstructed image (B, 3, 64, 64)
        """
        return self.decoder(z_q)
    
    def get_multi_one_hot(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to multi-one-hot representation.
        
        This is the key output for PPO training - a sparse binary vector
        representing the discrete visual state.
        
        Args:
            x: Input image (B, 3, 64, 64)
            
        Returns:
            multi_one_hot: Binary sparse vector (B, grid_size^2 * num_embeddings)
        """
        _, indices = self.encode(x)
        return self.indices_to_multi_one_hot(indices)
    
    def indices_to_multi_one_hot(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert codebook indices to multi-one-hot vector.
        
        Args:
            indices: Codebook indices (B, H, W)
            
        Returns:
            multi_one_hot: Binary sparse vector (B, H*W*num_embeddings)
        """
        batch_size = indices.shape[0]
        # Flatten spatial dimensions
        indices_flat = indices.view(batch_size, -1)  # (B, H*W)
        
        # Create one-hot for each position
        one_hots = F.one_hot(indices_flat, num_classes=self.num_embeddings)  # (B, H*W, K)
        
        # Flatten to single vector
        multi_one_hot = one_hots.view(batch_size, -1).float()  # (B, H*W*K)
        
        return multi_one_hot
    
    def get_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        """Get reconstruction without computing loss."""
        x_recon, _, _ = self.forward(x)
        return x_recon
    
    def compute_loss(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss for training.
        
        Args:
            x: Input image (B, 3, 64, 64)
            
        Returns:
            total_loss: Combined reconstruction + VQ loss
            loss_dict: Dictionary of individual losses
        """
        x_recon, _, vq_loss = self.forward(x)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)
        
        # Total loss
        total_loss = recon_loss + vq_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'vq_loss': vq_loss.item()
        }
        
        return total_loss, loss_dict


def create_vqvae(config: Optional[dict] = None) -> VQVAE:
    """
    Create VQ-VAE model from configuration.
    
    Args:
        config: Optional configuration dict
        
    Returns:
        Initialized VQ-VAE model
    """
    if config is None:
        config = {}
    
    vqvae_config = config.get('vqvae', {})
    
    return VQVAE(
        in_channels=3,
        embedding_dim=vqvae_config.get('embedding_dim', 64),
        num_embeddings=vqvae_config.get('codebook_size', 512),
        commitment_cost=vqvae_config.get('commitment_cost', 1.0),
        latent_grid_size=vqvae_config.get('latent_grid', 6)
    )


if __name__ == "__main__":
    # Quick test
    print("Testing VQ-VAE...")
    
    model = VQVAE()
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Multi-one-hot dimension: {model.multi_one_hot_dim}")
    
    # Test forward pass
    x = torch.randn(4, 3, 64, 64).clamp(0, 1)
    x_recon, indices, vq_loss = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    
    # Test multi-one-hot
    multi_one_hot = model.get_multi_one_hot(x)
    print(f"Multi-one-hot shape: {multi_one_hot.shape}")
    print(f"Multi-one-hot sparsity: {(multi_one_hot > 0).float().mean().item():.4f}")
    
    print("VQ-VAE test passed!")
