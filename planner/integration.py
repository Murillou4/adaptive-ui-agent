"""
LLM-RL Integration Loop

The complete pipeline:
User → LLM Planner → Plan → Translator → Reward Function → RL Training
"""

import os
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import yaml

from env.extended_sandbox import ExtendedSandboxEnv, make_extended_env
from planner.goal_dsl import StructuredPlan
from planner.llm_planner import LLMPlanner, create_planner, PlanResult
from planner.reward_generator import ObjectiveTranslator, DynamicRewardFunction
from vision.vqvae import VQVAE, create_vqvae
from agent.ppo import PPOAgent, create_ppo_agent


@dataclass
class IntegrationResult:
    """Result of LLM-RL integration training."""
    success: bool
    plan: Optional[StructuredPlan]
    episodes_trained: int
    final_success_rate: float
    final_progress: float
    message: str


class LLMRLIntegration:
    """
    Complete LLM-RL Integration System.
    
    Connects:
    - LLM Planner (intent → plan)
    - Objective Translator (plan → rewards)
    - RL Agent (rewards → actions)
    """
    
    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        llm_provider: str = "mock",
        device: str = "cpu"
    ):
        """
        Initialize integration system.
        
        Args:
            config_path: Path to configuration file
            llm_provider: LLM provider type ("mock", "openai", "ollama")
            device: Device for RL training
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = device
        
        # Create components
        self.env = make_extended_env()
        self.planner = create_planner(llm_provider)
        self.translator = ObjectiveTranslator(canvas_size=(64, 64))
        
        # VQ-VAE and Agent (lazy loaded)
        self.vqvae = None
        self.agent = None
        
        # Current task state
        self.current_plan: Optional[StructuredPlan] = None
        self.current_reward_fn: Optional[DynamicRewardFunction] = None
        
    def _ensure_models_loaded(self):
        """Ensure VQ-VAE and agent are loaded."""
        if self.vqvae is None:
            self.vqvae = create_vqvae(self.config).to(self.device)
            self.vqvae.eval()
        
        if self.agent is None:
            self.agent = create_ppo_agent(self.vqvae, self.config, self.device)
    
    def plan_from_request(self, user_request: str) -> PlanResult:
        """
        Generate plan from user request.
        
        Args:
            user_request: Natural language goal
            
        Returns:
            PlanResult with structured plan
        """
        result = self.planner.plan(user_request)
        
        if result.success:
            self.current_plan = result.plan
            self.current_reward_fn = self.translator.translate(result.plan)
            
            # Set dynamic reward in environment
            self.env.set_dynamic_reward(self.current_reward_fn.compute)
        
        return result
    
    def train_on_goal(
        self,
        user_request: str,
        max_episodes: int = 1000,
        success_threshold: float = 0.8,
        log_interval: int = 50
    ) -> IntegrationResult:
        """
        Full pipeline: plan and train on a user goal.
        
        Args:
            user_request: Natural language goal
            max_episodes: Maximum training episodes
            success_threshold: Success rate to consider done
            log_interval: Episodes between logging
            
        Returns:
            IntegrationResult with training outcome
        """
        # Step 1: Generate plan
        print(f"\n{'='*50}")
        print(f"Goal: {user_request}")
        print(f"{'='*50}")
        
        plan_result = self.plan_from_request(user_request)
        
        if not plan_result.success:
            return IntegrationResult(
                success=False,
                plan=None,
                episodes_trained=0,
                final_success_rate=0.0,
                final_progress=0.0,
                message=f"Planning failed: {plan_result.error}"
            )
        
        print(f"\n✅ Plan generated:")
        print(f"   Goal: {self.current_plan.goal}")
        print(f"   Elements: {len(self.current_plan.elements)}")
        print(f"   Constraints: {len(self.current_plan.constraints)}")
        
        # Step 2: Train RL agent
        self._ensure_models_loaded()
        
        print(f"\n🎯 Starting training (max {max_episodes} episodes)...")
        
        recent_successes = []
        best_progress = 0.0
        
        for episode in range(1, max_episodes + 1):
            # Run episode
            obs, info = self.env.reset()
            self.current_reward_fn.reset()
            
            episode_reward = 0
            
            while True:
                action, log_prob, value = self.agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                self.agent.store_transition(
                    obs, action, reward, next_obs,
                    terminated or truncated, log_prob, value
                )
                
                episode_reward += reward
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            # Check success
            success = self.current_reward_fn.is_success(obs)
            progress = self.current_reward_fn.get_progress(obs)
            
            recent_successes.append(float(success))
            if len(recent_successes) > 100:
                recent_successes.pop(0)
            
            best_progress = max(best_progress, progress)
            
            # PPO update
            if len(self.agent.buffer.states) >= 64:
                self.agent.update(obs)
            
            # Logging
            if episode % log_interval == 0:
                success_rate = np.mean(recent_successes)
                print(f"Episode {episode}: "
                      f"success_rate={success_rate:.1%}, "
                      f"progress={progress:.1%}, "
                      f"reward={episode_reward:.2f}")
                
                # Check if we've reached threshold
                if success_rate >= success_threshold:
                    print(f"\n🎉 Goal achieved! Success rate: {success_rate:.1%}")
                    return IntegrationResult(
                        success=True,
                        plan=self.current_plan,
                        episodes_trained=episode,
                        final_success_rate=success_rate,
                        final_progress=best_progress,
                        message="Training converged successfully"
                    )
        
        # Max episodes reached
        final_success_rate = np.mean(recent_successes)
        return IntegrationResult(
            success=final_success_rate >= success_threshold * 0.5,
            plan=self.current_plan,
            episodes_trained=max_episodes,
            final_success_rate=final_success_rate,
            final_progress=best_progress,
            message=f"Training completed after {max_episodes} episodes"
        )
    
    def run_interactive(self):
        """
        Run interactive mode.
        
        User can type goals and watch the agent learn.
        """
        print("\n" + "="*50)
        print("LLM-RL Integration - Interactive Mode")
        print("="*50)
        print("\nType a goal in natural language.")
        print("Examples:")
        print("  - 'Cria 3 quadrados azuis alinhados'")
        print("  - 'Cria um botão centralizado'")
        print("  - 'Cria um formulário simples'")
        print("\nType 'quit' to exit.")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n🎯 Goal: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                result = self.train_on_goal(
                    user_input,
                    max_episodes=500,
                    success_threshold=0.7,
                    log_interval=50
                )
                
                print(f"\n{'='*50}")
                print(f"Result: {'SUCCESS' if result.success else 'PARTIAL'}")
                print(f"Episodes: {result.episodes_trained}")
                print(f"Success Rate: {result.final_success_rate:.1%}")
                print(f"Best Progress: {result.final_progress:.1%}")
                print(f"{'='*50}")
                
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
    
    def evaluate_plan(self, obs: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate current observation against the plan.
        
        Args:
            obs: Current observation
            
        Returns:
            Evaluation metrics
        """
        if self.current_reward_fn is None:
            return {"error": "No plan set"}
        
        progress = self.current_reward_fn.get_progress(obs)
        success = self.current_reward_fn.is_success(obs)
        state = self.translator.detector.get_state_summary(obs)
        
        return {
            "progress": progress,
            "success": success,
            "state": state,
            "plan_goal": self.current_plan.goal if self.current_plan else None,
        }
    
    def close(self):
        """Clean up resources."""
        self.env.close()


def main():
    """Main entry point for LLM-RL integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-RL Integration")
    parser.add_argument('--config', '-c', default='configs/default.yaml')
    parser.add_argument('--provider', '-p', default='mock',
                        choices=['mock', 'openai', 'ollama'])
    parser.add_argument('--goal', '-g', type=str, default=None,
                        help='Single goal to train on')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    integration = LLMRLIntegration(
        config_path=args.config,
        llm_provider=args.provider
    )
    
    if args.goal:
        result = integration.train_on_goal(args.goal)
        print(f"\nFinal Result: {result.message}")
    elif args.interactive:
        integration.run_interactive()
    else:
        # Demo mode
        print("Running demo...")
        result = integration.train_on_goal(
            "Cria 3 quadrados azuis alinhados",
            max_episodes=200
        )
        print(f"\nDemo Result: {result.message}")
    
    integration.close()


if __name__ == "__main__":
    main()
