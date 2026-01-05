"""
Main Training Script for Adaptive UI Agent

Complete pipeline:
1. Generate dataset (if needed)
2. Pre-train VQ-VAE
3. Train PPO with VQ-VAE encoding
4. Launch chat interface
"""

import os
import sys
import argparse
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.sandbox_env import SandboxEnv
from env.dataset_generator import DatasetGenerator
from vision.vqvae import create_vqvae
from vision.train_vqvae import VQVAETrainer, ScreenshotDataset
from agent.trainer import Trainer
from interaction.chat_interface import ChatInterface


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def step_generate_dataset(config: dict, force: bool = False) -> str:
    """Step 1: Generate training dataset for VQ-VAE."""
    print("\n" + "=" * 50)
    print("STEP 1: Generating Dataset")
    print("=" * 50)
    
    training_config = config.get('training', {})
    dataset_config = config.get('dataset', {})
    
    dataset_dir = training_config.get('dataset_dir', 'data/screenshots')
    
    # Check if dataset already exists
    if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0 and not force:
        print(f"Dataset already exists at {dataset_dir}")
        print(f"Found {len(os.listdir(dataset_dir))} files")
        print("Use --force-dataset to regenerate")
        return dataset_dir
    
    generator = DatasetGenerator(
        output_dir=dataset_dir,
        num_samples=dataset_config.get('num_samples', 5000),
        include_variations=dataset_config.get('include_variations', True),
        noise_level=dataset_config.get('noise_level', 0.02)
    )
    
    return generator.generate()


def step_train_vqvae(config: dict, force: bool = False) -> str:
    """Step 2: Pre-train VQ-VAE on environment screenshots."""
    print("\n" + "=" * 50)
    print("STEP 2: Pre-training VQ-VAE")
    print("=" * 50)
    
    training_config = config.get('training', {})
    vqvae_config = config.get('vqvae', {})
    
    checkpoint_dir = training_config.get('checkpoint_dir', 'data/checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'vqvae_best.pt')
    
    # Check if already trained
    if os.path.exists(checkpoint_path) and not force:
        print(f"VQ-VAE checkpoint found at {checkpoint_path}")
        print("Use --force-vqvae to retrain")
        return checkpoint_path
    
    # Create model
    model = create_vqvae(config)
    
    # Create dataset
    dataset_dir = training_config.get('dataset_dir', 'data/screenshots')
    dataset = ScreenshotDataset(dataset_dir)
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=vqvae_config.get('batch_size', 64),
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
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
        checkpoint_dir=checkpoint_dir
    )
    
    # Train
    trainer.train(
        num_epochs=vqvae_config.get('num_epochs', 100),
        save_interval=10
    )
    
    return checkpoint_path


def step_train_ppo(config: dict, vqvae_path: str) -> str:
    """Step 3: Train PPO agent with VQ-VAE encoding."""
    print("\n" + "=" * 50)
    print("STEP 3: Training PPO Agent")
    print("=" * 50)
    
    training_config = config.get('training', {})
    ppo_config = config.get('ppo', {})
    
    # Create trainer
    trainer = Trainer(config_path="configs/default.yaml")
    
    # Load pre-trained VQ-VAE
    trainer.load_vqvae(vqvae_path)
    
    # Create agent
    trainer.create_agent()
    
    # Train
    metrics = trainer.train(
        num_episodes=training_config.get('total_episodes', 10000),
        steps_per_update=ppo_config.get('steps_per_update', 128),
        log_interval=training_config.get('log_interval', 100),
        save_interval=training_config.get('save_interval', 1000)
    )
    
    # Evaluate
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)
    
    eval_metrics = trainer.evaluate(num_episodes=100)
    print(f"Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
    print(f"Success Rate: {eval_metrics['success_rate']:.2%}")
    print(f"Mean Episode Length: {eval_metrics['mean_length']:.1f}")
    
    trainer.close()
    
    return os.path.join(
        training_config.get('checkpoint_dir', 'data/checkpoints'),
        'ppo_agent_latest.pt'
    )


def step_launch_chat(config: dict, agent_path: str):
    """Step 4: Launch interactive chat interface."""
    print("\n" + "=" * 50)
    print("STEP 4: Launching Chat Interface")
    print("=" * 50)
    
    # Create environment
    env = SandboxEnv()
    env.reset()
    
    # Load VQ-VAE
    vqvae = create_vqvae(config)
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'data/checkpoints')
    vqvae_path = os.path.join(checkpoint_dir, 'vqvae_best.pt')
    
    if os.path.exists(vqvae_path):
        checkpoint = torch.load(vqvae_path, map_location='cpu')
        vqvae.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded VQ-VAE from {vqvae_path}")
    
    # Create agent
    from agent.ppo import PPOAgent
    agent = PPOAgent(vqvae, vqvae.multi_one_hot_dim)
    
    if os.path.exists(agent_path):
        agent.load(agent_path)
        print(f"Loaded agent from {agent_path}")
    
    # Create and run chat interface
    chat = ChatInterface(agent=agent, env=env)
    chat.run_interactive()
    
    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Adaptive UI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training pipeline
  python scripts/run_training.py
  
  # Skip to PPO training (reuse existing VQ-VAE)
  python scripts/run_training.py --skip-vqvae
  
  # Only launch chat interface
  python scripts/run_training.py --chat-only
  
  # Force regenerate dataset
  python scripts/run_training.py --force-dataset
"""
    )
    
    parser.add_argument('--config', '-c', default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--skip-dataset', action='store_true',
                        help='Skip dataset generation')
    parser.add_argument('--skip-vqvae', action='store_true',
                        help='Skip VQ-VAE training')
    parser.add_argument('--skip-ppo', action='store_true',
                        help='Skip PPO training')
    parser.add_argument('--chat-only', action='store_true',
                        help='Only launch chat interface')
    parser.add_argument('--force-dataset', action='store_true',
                        help='Force regenerate dataset')
    parser.add_argument('--force-vqvae', action='store_true',
                        help='Force retrain VQ-VAE')
    parser.add_argument('--no-chat', action='store_true',
                        help='Do not launch chat after training')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("=" * 50)
    print("Adaptive UI Agent - Training Pipeline")
    print("Based on paper 2312.01203v3")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    checkpoint_dir = config.get('training', {}).get('checkpoint_dir', 'data/checkpoints')
    agent_path = os.path.join(checkpoint_dir, 'ppo_agent_latest.pt')
    
    if args.chat_only:
        step_launch_chat(config, agent_path)
        return
    
    # Step 1: Generate dataset
    if not args.skip_dataset:
        step_generate_dataset(config, force=args.force_dataset)
    
    # Step 2: Train VQ-VAE
    vqvae_path = os.path.join(checkpoint_dir, 'vqvae_best.pt')
    if not args.skip_vqvae:
        vqvae_path = step_train_vqvae(config, force=args.force_vqvae)
    
    # Step 3: Train PPO
    if not args.skip_ppo:
        agent_path = step_train_ppo(config, vqvae_path)
    
    # Step 4: Launch chat
    if not args.no_chat:
        step_launch_chat(config, agent_path)
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
