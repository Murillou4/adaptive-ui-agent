"""
Auto Training Script
Runs the agent through a curriculum of tasks, saving progress along the way.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from planner.integration import LLMRLIntegration
from training.curriculum_manager import CurriculumManager
from training.auto_curriculum import AutoCurriculum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("AutoTrain")


def main():
    parser = argparse.ArgumentParser(description="Auto Curriculum Training")
    parser.add_argument('--config', default='configs/default.yaml', help='Agent config')
    parser.add_argument('--curriculum', default='configs/initial_curriculum.yaml', help='Curriculum config')
    parser.add_argument('--provider', default='mock', choices=['mock', 'openai', 'ollama', 'google'], help='LLM provider')
    parser.add_argument('--max-episodes', type=int, default=500, help='Max episodes per task')
    parser.add_argument('--output-dir', default='checkpoints', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    logger.info(f"Initializing Curriculum Manager from {args.curriculum}")
    curriculum = CurriculumManager(args.curriculum)
    
    logger.info(f"Initializing Auto Curriculum (Provider: {args.provider})")
    # auto_curr = AutoCurriculum(curriculum) # Optional: enable if we want to generate tasks
    
    logger.info(f"Initializing Integration System (Provider: {args.provider})")
    integration = LLMRLIntegration(
        config_path=args.config,
        llm_provider=args.provider
    )
    
    total_tasks_completed = 0
    start_time = time.time()
    
    try:
        while True:
            # Get next task
            task = curriculum.get_next_task()
            
            if task is None:
                logger.info("No more tasks in curriculum! Training complete.")
                break
                
            logger.info(f"\n{'='*50}")
            logger.info(f"STARTING TASK: {task.name} (Level {curriculum.current_level_idx + 1})")
            logger.info(f"Goal: {task.goal_prompt}")
            logger.info(f"{'='*50}")
            
            # Train on task
            result = integration.train_on_goal(
                user_request=task.goal_prompt,
                max_episodes=args.max_episodes,
                success_threshold=0.8,
                log_interval=50
            )
            
            # Update curriculum
            curriculum.update_task_result(task.id, result.success)
            
            if result.success:
                logger.info(f"✅ Task Passed! (Success Rate: {result.final_success_rate:.1%})")
                total_tasks_completed += 1
            else:
                logger.info(f"❌ Task Failed. (Success Rate: {result.final_success_rate:.1%})")
            
            # Save checkpoint
            if integration.agent:
                checkpoint_path = os.path.join(args.output_dir, f"agent_lvl{curriculum.current_level_idx}_{task.id}.pt")
                integration.agent.save(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Also save "latest"
                latest_path = os.path.join(args.output_dir, "agent_latest.pt")
                integration.agent.save(latest_path)
            
            # Optional: Sleep briefly between tasks
            time.sleep(2)
            
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
    finally:
        integration.close()
        duration = time.time() - start_time
        logger.info(f"Training session ended. Completed {total_tasks_completed} tasks in {duration:.1f}s")

if __name__ == "__main__":
    main()
