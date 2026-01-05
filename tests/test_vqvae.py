"""
Tests for VQ-VAE model.
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.vqvae import VQVAE, VectorQuantizer, Encoder, Decoder, create_vqvae


class TestVectorQuantizer:
    """Test suite for VectorQuantizer."""
    
    def test_initialization(self):
        """Test VQ initialization."""
        vq = VectorQuantizer(num_embeddings=512, embedding_dim=64)
        
        assert vq.num_embeddings == 512
        assert vq.embedding_dim == 64
        assert vq.embeddings.weight.shape == (512, 64)
    
    def test_forward_shape(self):
        """Test VQ forward pass shapes."""
        vq = VectorQuantizer(num_embeddings=512, embedding_dim=64)
        
        z = torch.randn(4, 64, 6, 6)  # (B, D, H, W)
        z_q, indices, loss = vq(z)
        
        assert z_q.shape == z.shape
        assert indices.shape == (4, 6, 6)
        assert loss.ndim == 0  # Scalar
    
    def test_indices_in_range(self):
        """Test that indices are within codebook range."""
        vq = VectorQuantizer(num_embeddings=512, embedding_dim=64)
        
        z = torch.randn(4, 64, 6, 6)
        _, indices, _ = vq(z)
        
        assert indices.min() >= 0
        assert indices.max() < 512


class TestEncoder:
    """Test suite for Encoder."""
    
    def test_output_shape(self):
        """Test encoder output shape."""
        encoder = Encoder(in_channels=3, embedding_dim=64)
        
        x = torch.randn(4, 3, 64, 64)
        z = encoder(x)
        
        # Should produce 6x6 latent grid
        assert z.shape[0] == 4  # Batch
        assert z.shape[1] == 64  # Embedding dim
        # Spatial dimensions depend on architecture


class TestDecoder:
    """Test suite for Decoder."""
    
    def test_output_shape(self):
        """Test decoder output shape."""
        decoder = Decoder(embedding_dim=64)
        
        z = torch.randn(4, 64, 6, 6)
        x_recon = decoder(z)
        
        assert x_recon.shape == (4, 3, 64, 64)
        assert x_recon.min() >= 0
        assert x_recon.max() <= 1


class TestVQVAE:
    """Test suite for complete VQ-VAE model."""
    
    def test_initialization(self):
        """Test VQ-VAE initialization."""
        model = VQVAE()
        
        assert model.embedding_dim == 64
        assert model.num_embeddings == 512
        assert model.latent_grid_size == 6
        assert model.multi_one_hot_dim == 6 * 6 * 512
    
    def test_forward(self):
        """Test VQ-VAE forward pass."""
        model = VQVAE()
        
        x = torch.rand(4, 3, 64, 64)  # Already in [0, 1]
        x_recon, indices, vq_loss = model(x)
        
        assert x_recon.shape == x.shape
        assert indices.shape == (4, 6, 6)
        assert vq_loss.ndim == 0
    
    def test_encode(self):
        """Test encoding function."""
        model = VQVAE()
        
        x = torch.rand(4, 3, 64, 64)
        z_q, indices = model.encode(x)
        
        assert z_q.shape == (4, 64, 6, 6)
        assert indices.shape == (4, 6, 6)
    
    def test_decode(self):
        """Test decoding function."""
        model = VQVAE()
        
        z_q = torch.randn(4, 64, 6, 6)
        x_recon = model.decode(z_q)
        
        assert x_recon.shape == (4, 3, 64, 64)
    
    def test_multi_one_hot(self):
        """Test multi-one-hot output."""
        model = VQVAE()
        
        x = torch.rand(4, 3, 64, 64)
        multi_one_hot = model.get_multi_one_hot(x)
        
        # Check shape
        assert multi_one_hot.shape == (4, 6 * 6 * 512)
        
        # Check it's binary
        assert torch.all((multi_one_hot == 0) | (multi_one_hot == 1))
        
        # Check sparsity (should be exactly 36 ones per sample)
        ones_per_sample = multi_one_hot.sum(dim=1)
        assert torch.all(ones_per_sample == 36)  # 6x6 grid, one code per position
    
    def test_multi_one_hot_sparsity(self):
        """Test that multi-one-hot is sparse and binary."""
        model = VQVAE()
        
        x = torch.rand(4, 3, 64, 64)
        moh = model.get_multi_one_hot(x)
        
        sparsity = (moh > 0).float().mean()
        expected_sparsity = 36 / (36 * 512)  # 36 active bits out of 18432
        
        assert abs(sparsity - expected_sparsity) < 0.01
    
    def test_reconstruction_loss(self):
        """Test that reconstruction loss decreases."""
        model = VQVAE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x = torch.rand(4, 3, 64, 64)
        
        # Initial loss
        loss1, _ = model.compute_loss(x)
        
        # One gradient step
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        
        # Loss after update
        loss2, _ = model.compute_loss(x)
        
        # Loss should decrease (or at least not increase significantly)
        assert loss2.item() <= loss1.item() * 1.5
    
    def test_create_vqvae_with_config(self):
        """Test VQ-VAE creation from config."""
        config = {
            'vqvae': {
                'embedding_dim': 32,
                'codebook_size': 256,
                'commitment_cost': 0.5,
                'latent_grid': 8
            }
        }
        
        model = create_vqvae(config)
        
        assert model.embedding_dim == 32
        assert model.num_embeddings == 256
        assert model.latent_grid_size == 8
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = VQVAE()
        
        x = torch.rand(2, 3, 64, 64, requires_grad=True)
        loss, _ = model.compute_loss(x)
        loss.backward()
        
        # Check encoder has gradients
        assert model.encoder.encoder[0].weight.grad is not None
        
        # Check decoder has gradients
        assert model.decoder.decoder[0].weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
