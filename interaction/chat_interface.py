"""
Chat Interface for Agent Control

A text-based control panel for the Adaptive UI Agent.
Not a chatbot - a command interface for observing and controlling the agent.
"""

import os
import re
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np

from env.sandbox_env import SandboxEnv
from agent.ppo import PPOAgent


@dataclass
class CommandResult:
    """Result of a command execution."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    image_path: Optional[str] = None


class ChatInterface:
    """
    Command-based interface for controlling and inspecting the agent.
    
    Supported commands:
    - status: Show agent state and training progress
    - set_rule <rule>: Change environment rule (blue_bad, blue_good)
    - set_reward <type> <value>: Modify reward values
    - freeze_encoder: Freeze VQ-VAE weights
    - unfreeze_encoder: Unfreeze VQ-VAE weights
    - show_latent: Display multi-one-hot visualization
    - swap_targets: Trigger continual RL adaptation
    - screenshot: Capture current environment state
    - reconstruct: Show VQ-VAE reconstruction
    - step <action>: Take a single step with given action
    - reset: Reset the environment
    - pause / resume: Control training
    - help: Show available commands
    """
    
    def __init__(
        self,
        agent: Optional[PPOAgent] = None,
        env: Optional[SandboxEnv] = None,
        output_dir: str = "data/chat_output"
    ):
        """
        Initialize chat interface.
        
        Args:
            agent: PPO agent instance
            env: Environment instance
            output_dir: Directory for saving visualizations
        """
        self.agent = agent
        self.env = env
        self.output_dir = output_dir
        self.paused = False
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Command registry
        self.commands: Dict[str, Callable] = {
            'help': self._cmd_help,
            'status': self._cmd_status,
            'set_rule': self._cmd_set_rule,
            'set_reward': self._cmd_set_reward,
            'freeze_encoder': self._cmd_freeze_encoder,
            'unfreeze_encoder': self._cmd_unfreeze_encoder,
            'show_latent': self._cmd_show_latent,
            'swap_targets': self._cmd_swap_targets,
            'screenshot': self._cmd_screenshot,
            'reconstruct': self._cmd_reconstruct,
            'step': self._cmd_step,
            'reset': self._cmd_reset,
            'pause': self._cmd_pause,
            'resume': self._cmd_resume,
            'metrics': self._cmd_metrics,
        }
        
        # Current observation (cached)
        self._current_obs = None
        
    def parse_command(self, command: str) -> CommandResult:
        """
        Parse and execute a command.
        
        Args:
            command: Command string from user
            
        Returns:
            Command result with message and optional data
        """
        command = command.strip()
        if not command:
            return CommandResult(False, "Empty command. Type 'help' for available commands.")
        
        parts = command.split()
        cmd_name = parts[0].lower()
        args = parts[1:]
        
        if cmd_name not in self.commands:
            return CommandResult(
                False, 
                f"Unknown command: '{cmd_name}'. Type 'help' for available commands."
            )
        
        try:
            return self.commands[cmd_name](*args)
        except Exception as e:
            return CommandResult(False, f"Error executing '{cmd_name}': {str(e)}")
    
    # ==================== Command Implementations ====================
    
    def _cmd_help(self, *args) -> CommandResult:
        """Show help message."""
        help_text = """
Available Commands:
==================
status              - Show agent state and training progress
set_rule <rule>     - Change rule (blue_bad, blue_good)
set_reward <t> <v>  - Set reward: click_target, click_obstacle, click_background, step_penalty
freeze_encoder      - Freeze VQ-VAE encoder weights
unfreeze_encoder    - Unfreeze VQ-VAE encoder
show_latent         - Display multi-one-hot visualization
swap_targets        - Swap target/obstacle roles (continual RL)
screenshot          - Capture current environment
reconstruct         - Show VQ-VAE reconstruction
step <action>       - Take step (0-7: move, 8: click)
reset               - Reset environment
pause               - Pause training
resume              - Resume training
metrics             - Show training metrics
help                - Show this message
"""
        return CommandResult(True, help_text.strip())
    
    def _cmd_status(self, *args) -> CommandResult:
        """Show agent and environment status."""
        status = {
            'environment': {},
            'agent': {},
            'training': {'paused': self.paused}
        }
        
        if self.env:
            info = self.env._get_info()
            status['environment'] = {
                'step_count': info['step_count'],
                'total_reward': info['total_reward'],
                'cursor_pos': info['cursor_pos'],
                'target_is_blue': info['target_is_blue'],
                'target_pos': info['target_pos'],
                'obstacle_pos': info['obstacle_pos']
            }
        
        if self.agent:
            status['agent'] = {
                'encoder_frozen': self.agent.encoder_frozen,
                'device': str(self.agent.device),
                'buffer_size': len(self.agent.buffer.states)
            }
        
        msg = "=== Agent Status ===\n"
        for category, data in status.items():
            msg += f"\n{category.upper()}:\n"
            for key, value in data.items():
                msg += f"  {key}: {value}\n"
        
        return CommandResult(True, msg.strip(), data=status)
    
    def _cmd_set_rule(self, *args) -> CommandResult:
        """Set environment rule."""
        if not self.env:
            return CommandResult(False, "Environment not initialized")
        
        if not args:
            return CommandResult(False, "Usage: set_rule <rule> (blue_bad, blue_good)")
        
        rule = args[0].lower()
        try:
            self.env.set_rule(rule)
            return CommandResult(True, f"Rule set to: {rule}")
        except ValueError as e:
            return CommandResult(False, str(e))
    
    def _cmd_set_reward(self, *args) -> CommandResult:
        """Set a reward value."""
        if not self.env:
            return CommandResult(False, "Environment not initialized")
        
        if len(args) < 2:
            return CommandResult(
                False, 
                "Usage: set_reward <type> <value>\n"
                "Types: click_target, click_obstacle, click_background, step_penalty"
            )
        
        reward_type = args[0]
        try:
            value = float(args[1])
            self.env.set_reward(reward_type, value)
            return CommandResult(True, f"Reward '{reward_type}' set to {value}")
        except ValueError as e:
            return CommandResult(False, str(e))
    
    def _cmd_freeze_encoder(self, *args) -> CommandResult:
        """Freeze VQ-VAE encoder."""
        if not self.agent:
            return CommandResult(False, "Agent not initialized")
        
        self.agent.freeze_encoder()
        return CommandResult(True, "VQ-VAE encoder frozen")
    
    def _cmd_unfreeze_encoder(self, *args) -> CommandResult:
        """Unfreeze VQ-VAE encoder."""
        if not self.agent:
            return CommandResult(False, "Agent not initialized")
        
        self.agent.unfreeze_encoder()
        return CommandResult(True, "VQ-VAE encoder unfrozen")
    
    def _cmd_show_latent(self, *args) -> CommandResult:
        """Show multi-one-hot representation."""
        if not self.agent or not self.env:
            return CommandResult(False, "Agent and environment must be initialized")
        
        # Get current observation
        obs = self.env.get_screenshot()
        
        # Encode to multi-one-hot
        multi_one_hot = self.agent.encode_observation(obs)
        moh_np = multi_one_hot.cpu().numpy().squeeze()
        
        # Get statistics
        active_bits = np.sum(moh_np > 0)
        total_bits = len(moh_np)
        sparsity = active_bits / total_bits
        
        # Reshape to grid for visualization
        grid_size = int(np.sqrt(len(moh_np) // self.agent.vqvae.num_embeddings))
        
        msg = f"""
=== Multi-One-Hot Representation ===
Total dimension: {total_bits}
Active bits: {active_bits}
Sparsity: {sparsity:.4f}
Grid size: {grid_size}x{grid_size}
Codebook size: {self.agent.vqvae.num_embeddings}
"""
        
        return CommandResult(True, msg.strip(), data={
            'multi_one_hot': moh_np.tolist(),
            'active_bits': int(active_bits),
            'sparsity': float(sparsity)
        })
    
    def _cmd_swap_targets(self, *args) -> CommandResult:
        """Swap target and obstacle roles."""
        if not self.env:
            return CommandResult(False, "Environment not initialized")
        
        old_state = "blue" if self.env.target_is_blue else "red"
        self.env.swap_targets()
        new_state = "blue" if self.env.target_is_blue else "red"
        
        return CommandResult(
            True, 
            f"Targets swapped! Target color: {old_state} -> {new_state}"
        )
    
    def _cmd_screenshot(self, *args) -> CommandResult:
        """Capture and save environment screenshot."""
        if not self.env:
            return CommandResult(False, "Environment not initialized")
        
        obs = self.env.get_screenshot()
        
        from PIL import Image
        img = Image.fromarray(obs)
        path = os.path.join(self.output_dir, "screenshot.png")
        img.save(path)
        
        return CommandResult(
            True, 
            f"Screenshot saved to {path}",
            image_path=path
        )
    
    def _cmd_reconstruct(self, *args) -> CommandResult:
        """Show VQ-VAE reconstruction."""
        if not self.agent or not self.env:
            return CommandResult(False, "Agent and environment must be initialized")
        
        import torch
        from PIL import Image
        
        # Get observation
        obs = self.env.get_screenshot()
        
        # Convert to tensor
        obs_tensor = torch.from_numpy(obs).float() / 255.0
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
        obs_tensor = obs_tensor.to(self.agent.device)
        
        # Get reconstruction
        with torch.no_grad():
            recon = self.agent.vqvae.get_reconstruction(obs_tensor)
        
        # Save both
        recon_np = (recon.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        orig_path = os.path.join(self.output_dir, "original.png")
        recon_path = os.path.join(self.output_dir, "reconstruction.png")
        
        Image.fromarray(obs).save(orig_path)
        Image.fromarray(recon_np).save(recon_path)
        
        # Compute reconstruction error
        mse = np.mean((obs.astype(float) - recon_np.astype(float)) ** 2)
        
        return CommandResult(
            True,
            f"Original: {orig_path}\nReconstruction: {recon_path}\nMSE: {mse:.2f}",
            data={'mse': float(mse)},
            image_path=recon_path
        )
    
    def _cmd_step(self, *args) -> CommandResult:
        """Take a single step with given action."""
        if not self.env:
            return CommandResult(False, "Environment not initialized")
        
        if not args:
            return CommandResult(False, "Usage: step <action> (0-8)")
        
        try:
            action = int(args[0])
            if not 0 <= action <= 8:
                return CommandResult(False, "Action must be 0-8")
        except ValueError:
            return CommandResult(False, "Action must be an integer")
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_obs = obs
        
        action_names = ['up', 'down', 'left', 'right', 'up-left', 'up-right', 'down-left', 'down-right', 'click']
        
        msg = f"""
Action: {action} ({action_names[action]})
Reward: {reward:.2f}
Terminated: {terminated}
Truncated: {truncated}
Cursor: {info['cursor_pos']}
"""
        
        return CommandResult(True, msg.strip(), data={
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'info': info
        })
    
    def _cmd_reset(self, *args) -> CommandResult:
        """Reset the environment."""
        if not self.env:
            return CommandResult(False, "Environment not initialized")
        
        obs, info = self.env.reset()
        self._current_obs = obs
        
        return CommandResult(
            True, 
            f"Environment reset. Cursor at {info['cursor_pos']}",
            data=info
        )
    
    def _cmd_pause(self, *args) -> CommandResult:
        """Pause training."""
        self.paused = True
        return CommandResult(True, "Training paused")
    
    def _cmd_resume(self, *args) -> CommandResult:
        """Resume training."""
        self.paused = False
        return CommandResult(True, "Training resumed")
    
    def _cmd_metrics(self, *args) -> CommandResult:
        """Show training metrics."""
        if not self.agent:
            return CommandResult(False, "Agent not initialized")
        
        msg = """
=== Training Metrics ===
Buffer size: {buffer_size}
Encoder frozen: {frozen}
""".format(
            buffer_size=len(self.agent.buffer.states),
            frozen=self.agent.encoder_frozen
        )
        
        return CommandResult(True, msg.strip())
    
    # ==================== Interactive Loop ====================
    
    def run_interactive(self):
        """Run interactive command loop."""
        print("=" * 50)
        print("Adaptive UI Agent - Chat Interface")
        print("Type 'help' for available commands, 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                result = self.parse_command(command)
                
                if result.success:
                    print(result.message)
                else:
                    print(f"Error: {result.message}")
                    
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
            except EOFError:
                break


def create_chat_interface(
    agent: Optional[PPOAgent] = None,
    env: Optional[SandboxEnv] = None
) -> ChatInterface:
    """Create a chat interface instance."""
    return ChatInterface(agent=agent, env=env)


if __name__ == "__main__":
    # Quick test without agent
    from env.sandbox_env import SandboxEnv
    
    env = SandboxEnv()
    env.reset()
    
    chat = ChatInterface(env=env)
    
    # Test some commands
    print(chat.parse_command("help").message)
    print(chat.parse_command("status").message)
    print(chat.parse_command("reset").message)
    print(chat.parse_command("step 3").message)
    print(chat.parse_command("screenshot").message)
    
    env.close()
