"""
Interactive Demo for Adaptive UI Agent

Load a trained agent and run in inference mode with visualization.
"""

import os
import sys
import argparse
import time
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.sandbox_env import SandboxEnv
from vision.vqvae import create_vqvae
from agent.ppo import PPOAgent
from interaction.chat_interface import ChatInterface
from interaction.visualizer import Visualizer


def load_agent(config_path: str, vqvae_path: str, agent_path: str) -> tuple:
    """Load trained VQ-VAE and PPO agent."""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load VQ-VAE
    vqvae = create_vqvae(config)
    if os.path.exists(vqvae_path):
        checkpoint = torch.load(vqvae_path, map_location=device)
        vqvae.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded VQ-VAE from {vqvae_path}")
    else:
        print(f"Warning: VQ-VAE checkpoint not found at {vqvae_path}")
    
    vqvae = vqvae.to(device)
    vqvae.eval()
    
    # Create agent
    agent = PPOAgent(vqvae, vqvae.multi_one_hot_dim, device=device)
    
    if os.path.exists(agent_path):
        agent.load(agent_path)
        print(f"Loaded agent from {agent_path}")
    else:
        print(f"Warning: Agent checkpoint not found at {agent_path}")
    
    return vqvae, agent, config


def run_demo_episode(env: SandboxEnv, agent: PPOAgent, visualizer: Visualizer, 
                     verbose: bool = True, delay: float = 0.1) -> dict:
    """
    Run a single demo episode with visualization.
    
    Args:
        env: Environment
        agent: Trained agent
        visualizer: Visualization tools
        verbose: Print step information
        delay: Delay between steps (seconds)
        
    Returns:
        Episode statistics
    """
    obs, info = env.reset()
    
    episode_reward = 0
    steps = 0
    success = False
    click_positions = []
    
    action_names = ['↑ Up', '↓ Down', '← Left', '→ Right',
                   '↖ Up-Left', '↗ Up-Right', '↙ Down-Left', '↘ Down-Right',
                   '● Click']
    
    if verbose:
        print("\n" + "=" * 40)
        print("Starting Demo Episode")
        print("=" * 40)
    
    while True:
        # Get action
        action, log_prob, value = agent.get_action(obs)
        
        if verbose:
            print(f"\nStep {steps + 1}:")
            print(f"  Cursor: {info['cursor_pos']}")
            print(f"  Action: {action_names[action]}")
            print(f"  Value: {value:.3f}")
        
        # Track click positions
        if action == 8:
            click_positions.append(tuple(env.state.cursor_pos))
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        steps += 1
        
        if verbose:
            print(f"  Reward: {reward:+.2f}")
        
        if terminated:
            if reward > 0:
                success = True
                if verbose:
                    print("\n🎯 SUCCESS! Clicked on target!")
            else:
                if verbose:
                    print("\n❌ FAILED! Clicked on obstacle!")
        
        if terminated or truncated:
            break
        
        if delay > 0:
            time.sleep(delay)
    
    if verbose:
        print("\n" + "-" * 40)
        print(f"Episode Complete!")
        print(f"  Steps: {steps}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Success: {success}")
        print("-" * 40)
    
    return {
        'steps': steps,
        'reward': episode_reward,
        'success': success,
        'click_positions': click_positions
    }


def run_interactive_demo(env: SandboxEnv, agent: PPOAgent, 
                         visualizer: Visualizer, chat: ChatInterface):
    """
    Run interactive demo with chat commands.
    """
    print("\n" + "=" * 50)
    print("Adaptive UI Agent - Interactive Demo")
    print("=" * 50)
    print("\nCommands:")
    print("  run       - Run a demo episode")
    print("  run N     - Run N demo episodes")
    print("  step      - Single step with agent action")
    print("  dashboard - Generate agent dashboard")
    print("  quit      - Exit demo")
    print("\nYou can also use any chat command (type 'help' for list)")
    print("=" * 50)
    
    obs, info = env.reset()
    
    while True:
        try:
            command = input("\ndemo> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if command.lower() == 'run':
                result = run_demo_episode(env, agent, visualizer)
                obs, info = env.reset()
                
            elif command.lower().startswith('run '):
                try:
                    n = int(command.split()[1])
                    successes = 0
                    total_reward = 0
                    
                    for i in range(n):
                        print(f"\n--- Episode {i+1}/{n} ---")
                        result = run_demo_episode(env, agent, visualizer, 
                                                 verbose=False, delay=0)
                        successes += int(result['success'])
                        total_reward += result['reward']
                        print(f"Steps: {result['steps']}, "
                              f"Reward: {result['reward']:.2f}, "
                              f"Success: {result['success']}")
                    
                    print(f"\n=== Summary ===")
                    print(f"Success Rate: {successes}/{n} ({successes/n:.1%})")
                    print(f"Mean Reward: {total_reward/n:.2f}")
                    
                except ValueError:
                    print("Usage: run N (where N is a number)")
                    
            elif command.lower() == 'step':
                action, log_prob, value = agent.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                
                action_names = ['↑', '↓', '←', '→', '↖', '↗', '↙', '↘', '●']
                print(f"Action: {action_names[action]}, Reward: {reward:+.2f}, "
                      f"Cursor: {info['cursor_pos']}")
                
                if terminated or truncated:
                    print("Episode ended. Resetting...")
                    obs, info = env.reset()
                    
            elif command.lower() == 'dashboard':
                # Generate comprehensive dashboard
                obs_current = env.get_screenshot()
                
                # Get reconstruction
                obs_tensor = torch.from_numpy(obs_current).float() / 255.0
                obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
                obs_tensor = obs_tensor.to(agent.device)
                
                with torch.no_grad():
                    recon = agent.vqvae.get_reconstruction(obs_tensor)
                    multi_one_hot = agent.vqvae.get_multi_one_hot(obs_tensor)
                    
                    # Get action probs
                    state = agent.encode_observation(obs_current)
                    dist = agent.policy.get_action_distribution(state)
                    action_probs = dist.probs.cpu().numpy().squeeze()
                
                recon_np = (recon.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                moh_np = multi_one_hot.cpu().numpy().squeeze()
                
                path = visualizer.create_agent_dashboard(
                    obs_current, recon_np, moh_np, action_probs, info
                )
                print(f"Dashboard saved to: {path}")
                
            else:
                # Try as chat command
                result = chat.parse_command(command)
                print(result.message)
                
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(description="Interactive Demo for Adaptive UI Agent")
    
    parser.add_argument('--config', '-c', default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--vqvae', default='data/checkpoints/vqvae_best.pt',
                        help='Path to VQ-VAE checkpoint')
    parser.add_argument('--agent', default='data/checkpoints/ppo_agent_latest.pt',
                        help='Path to agent checkpoint')
    parser.add_argument('--episodes', '-n', type=int, default=None,
                        help='Run N episodes and exit (non-interactive)')
    parser.add_argument('--no-delay', action='store_true',
                        help='Disable step delay in visualization')
    
    args = parser.parse_args()
    
    # Load agent
    print("Loading trained agent...")
    vqvae, agent, config = load_agent(args.config, args.vqvae, args.agent)
    
    # Create environment
    env = SandboxEnv()
    
    # Create visualizer
    visualizer = Visualizer()
    
    # Create chat interface
    chat = ChatInterface(agent=agent, env=env)
    
    if args.episodes:
        # Non-interactive mode
        successes = 0
        rewards = []
        
        for i in range(args.episodes):
            result = run_demo_episode(
                env, agent, visualizer,
                verbose=False,
                delay=0 if args.no_delay else 0.05
            )
            successes += int(result['success'])
            rewards.append(result['reward'])
            
            print(f"Episode {i+1}: reward={result['reward']:.2f}, "
                  f"success={result['success']}")
        
        print(f"\n=== Final Results ===")
        print(f"Success Rate: {successes/args.episodes:.1%}")
        print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        
    else:
        # Interactive mode
        run_interactive_demo(env, agent, visualizer, chat)
    
    env.close()


if __name__ == "__main__":
    main()
