"""
Dataset Generator for VQ-VAE Training
Generates screenshots from the sandbox environment with variations.
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
import yaml

from env.sandbox_env import SandboxEnv, EnvConfig


class DatasetGenerator:
    """
    Generates training dataset for VQ-VAE pre-training.
    
    Creates screenshots with:
    - Random target/obstacle positions
    - Color variations (within color ranges)
    - Light noise augmentation
    """
    
    def __init__(
        self,
        output_dir: str = "data/screenshots",
        num_samples: int = 5000,
        include_variations: bool = True,
        noise_level: float = 0.02,
        seed: Optional[int] = 42
    ):
        """
        Initialize the dataset generator.
        
        Args:
            output_dir: Directory to save screenshots
            num_samples: Number of samples to generate
            include_variations: Whether to include color/position variations
            noise_level: Amount of noise to add (0-1)
            seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.include_variations = include_variations
        self.noise_level = noise_level
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            
    def generate(self) -> str:
        """
        Generate the dataset.
        
        Returns:
            Path to the generated dataset
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create environment
        env = SandboxEnv()
        
        print(f"Generating {self.num_samples} screenshots...")
        
        for i in tqdm(range(self.num_samples)):
            # Reset environment (randomizes positions)
            env.reset(seed=self.seed + i if self.seed else None)
            
            # Optionally move cursor to random position
            if self.include_variations:
                cursor_x = np.random.randint(0, env.config.size)
                cursor_y = np.random.randint(0, env.config.size)
                env.state.cursor_pos = np.array([cursor_x, cursor_y])
            
            # Get screenshot
            screenshot = env.get_screenshot()
            
            # Add noise if enabled
            if self.include_variations and self.noise_level > 0:
                screenshot = self._add_noise(screenshot)
            
            # Save image
            img = Image.fromarray(screenshot)
            img.save(os.path.join(self.output_dir, f"sample_{i:05d}.png"))
            
            # Generate variant with swapped colors (for continual RL)
            if self.include_variations and i < self.num_samples // 2:
                env.swap_targets()
                variant = env.get_screenshot()
                if self.noise_level > 0:
                    variant = self._add_noise(variant)
                img_variant = Image.fromarray(variant)
                img_variant.save(os.path.join(self.output_dir, f"sample_{i:05d}_swap.png"))
                env.swap_targets()  # Reset
        
        env.close()
        
        print(f"Dataset saved to {self.output_dir}")
        return self.output_dir
    
    def _add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add light Gaussian noise to image."""
        noise = np.random.randn(*image.shape) * self.noise_level * 255
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def get_dataset_stats(self) -> dict:
        """Get statistics about the generated dataset."""
        if not os.path.exists(self.output_dir):
            return {"error": "Dataset not found"}
            
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        
        return {
            "num_samples": len(files),
            "output_dir": self.output_dir,
            "file_format": "PNG",
            "image_size": "64x64"
        }


def generate_from_config(config_path: str = "configs/default.yaml") -> str:
    """Generate dataset using configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config.get('dataset', {})
    
    generator = DatasetGenerator(
        output_dir=config.get('training', {}).get('dataset_dir', 'data/screenshots'),
        num_samples=dataset_config.get('num_samples', 5000),
        include_variations=dataset_config.get('include_variations', True),
        noise_level=dataset_config.get('noise_level', 0.02)
    )
    
    return generator.generate()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate VQ-VAE training dataset")
    parser.add_argument('--output', '-o', default='data/screenshots', 
                        help='Output directory')
    parser.add_argument('--num-samples', '-n', type=int, default=5000,
                        help='Number of samples to generate')
    parser.add_argument('--no-variations', action='store_true',
                        help='Disable color/position variations')
    parser.add_argument('--noise', type=float, default=0.02,
                        help='Noise level (0-1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Use config file instead of arguments')
    
    args = parser.parse_args()
    
    if args.config:
        generate_from_config(args.config)
    else:
        generator = DatasetGenerator(
            output_dir=args.output,
            num_samples=args.num_samples,
            include_variations=not args.no_variations,
            noise_level=args.noise,
            seed=args.seed
        )
        generator.generate()
        print(f"\nDataset stats: {generator.get_dataset_stats()}")
