"""
Adaptive UI Agent - Universal Demo

This script demonstrates the Universal Adaptive UI Agent in action.
It uses the Hierarchical Controller to interact with the current screen.

Usage:
    python run_agent.py --goal "Open notepad and type hello"

Safety:
    - Move mouse to top-left corner to abort (PyAutoGUI failsafe)
"""

import argparse
import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepMind-Agent")

from env.universal_env import UniversalEnv
from controller.meta_controller import MetaController

async def main():
    parser = argparse.ArgumentParser(description="Run Adaptive UI Agent")
    parser.add_argument("--goal", type=str, default="Search for cats on Google", help="Goal to achieve")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index")
    args = parser.parse_args()

    print(f"🚀 Starting Agent with goal: '{args.goal}'")
    print("⚠️  SAFETY: Move mouse quickly to a corner to abort!")
    
    # Initialize Environment
    # We use a smaller resolution for vision speed, but capturing full screen
    env = UniversalEnv(target_resolution=(640, 480))
    
    # Initialize Controller
    controller = MetaController()
    
    try:
        # Run goal
        result = await controller.achieve_goal(args.goal, env)
        
        if result.success:
            print("\n✅ Goal Achieved successfully!")
            print(f"Steps: {result.steps_executed}")
            print(f"Skills: {', '.join(result.executed_skills)}")
        else:
            print(f"\n❌ Goal Failed: {result.message}")
            
    except KeyboardInterrupt:
        print("\n🛑 Aborted by user")
    except Exception as e:
        logger.exception("Agent crashed")
    finally:
        env.close()

if __name__ == "__main__":
    asyncio.run(main())
