"""
train_hmarl.py

Example training script for HMARL system.

Demonstrates:
- Agent initialization with LearningWrapper
- Centralized Training, Decentralized Execution (CTDE)
- Parameter sharing across store agents
- Integration with digital twin and reconciliation engine
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.store_agent import StoreAgent
from agents.warehouse_agent import WarehouseAgent
from agents.supplier_agent import SupplierAgent
from agents.learning_wrapper import LearningWrapper
from agents.ppo_trainer import PPOTrainer
from env.digital_twin import DigitalTwin
import config.simulation_config as config


def create_agents(ppo_trainer=None, learning_enabled=False):
    """
    Create all agents for the HMARL system.
    
    Args:
        ppo_trainer: Optional PPO trainer for store agents
        learning_enabled: Whether to enable learning for store agents
    
    Returns:
        Dict of agent_id -> LearningWrapper/Agent
    """
    agents = {}
    
    # Store Agent 1 (learning-enabled)
    store_agent_1 = StoreAgent(
        agent_id='store_agent_1',
        store_id='store_1',
        config={'reorder_point': 300, 'order_up_to_level': 700}
    )
    agents['store_1'] = LearningWrapper(
        agent=store_agent_1,
        learner=ppo_trainer,
        learning_enabled=learning_enabled
    )
    
    # Store Agent 2 (learning-enabled, shares policy with Store Agent 1)
    store_agent_2 = StoreAgent(
        agent_id='store_agent_2',
        store_id='store_2',
        config={'reorder_point': 300, 'order_up_to_level': 700}
    )
    agents['store_2'] = LearningWrapper(
        agent=store_agent_2,
        learner=ppo_trainer,  # Same learner = shared policy
        learning_enabled=learning_enabled
    )
    
    # Warehouse Agent (rule-based for now)
    warehouse_agent = WarehouseAgent(
        agent_id='warehouse_agent',
        warehouse_id='warehouse_1',
        config={'reorder_point': 2000, 'order_up_to_level': 4000}
    )
    agents['warehouse_1'] = warehouse_agent
    
    # Supplier Agent (rule-based)
    supplier_agent = SupplierAgent(
        agent_id='supplier_agent',
        supplier_id='supplier_1',
        config={'production_capacity': 10000, 'lead_time': 7}
    )
    agents['supplier_1'] = supplier_agent
    
    return agents


def simulate_reconciliation(state, actions, next_state):
    """
    Simulate reconciliation engine output.
    
    In production, this would call the actual reconciliation engine.
    For now, we compute basic metrics from state.
    
    Args:
        state: Current state
        actions: Actions taken
        next_state: Next state
    
    Returns:
        Dict of agent_id -> reconciliation_report
    """
    reports = {}
    
    # Store 1 report
    store_1_state = next_state['stores'].get('store_1', {})
    lost_sales = sum(store_1_state.get('daily_lost_sales', {}).values())
    inventory = sum(store_1_state.get('inventory', {}).values())
    
    reports['store_1'] = {
        'service_level': 1.0 if lost_sales == 0 else 0.9,
        'holding_cost': inventory * 0.1,
        'stockout_penalty': lost_sales * 10.0,
        'excess_inventory': max(0, inventory - 500)
    }
    
    # Store 2 report
    store_2_state = next_state['stores'].get('store_2', {})
    lost_sales_2 = sum(store_2_state.get('daily_lost_sales', {}).values())
    inventory_2 = sum(store_2_state.get('inventory', {}).values())
    
    reports['store_2'] = {
        'service_level': 1.0 if lost_sales_2 == 0 else 0.9,
        'holding_cost': inventory_2 * 0.1,
        'stockout_penalty': lost_sales_2 * 10.0,
        'excess_inventory': max(0, inventory_2 - 500)
    }
    
    # Warehouse report
    warehouse_state = list(next_state['warehouses'].values())[0] if next_state['warehouses'] else {}
    warehouse_inv = sum(warehouse_state.get('inventory', {}).values())
    
    reports['warehouse_1'] = {
        'avg_store_service_level': (reports['store_1']['service_level'] + reports['store_2']['service_level']) / 2,
        'holding_cost': warehouse_inv * 0.05,
        'store_stockout_count': (1 if lost_sales > 0 else 0) + (1 if lost_sales_2 > 0 else 0)
    }
    
    # Supplier report
    reports['supplier_1'] = {
        'fulfillment_rate': 1.0,
        'delay_penalty': 0.0
    }
    
    return reports


def train_episode(env, agents, num_days=30):
    """
    Run one training episode.
    
    Args:
        env: Digital twin environment
        agents: Dict of agents
        num_days: Number of days to simulate
    
    Returns:
        Episode statistics
    """
    state = env.reset()
    episode_rewards = {agent_id: [] for agent_id in agents.keys()}
    
    for day in range(num_days):
        # 1. OBSERVE: All agents observe state
        observations = {}
        for agent_id, agent in agents.items():
            observations[agent_id] = agent.observe(state)
        
        # 2. ACT: All agents select actions
        actions = {}
        for agent_id, agent in agents.items():
            actions[agent_id] = agent.act(observations[agent_id])
        
        # 3. EXECUTE: Environment steps forward
        next_state, _, _, _ = env.step(actions)
        
        # 4. RECONCILE: Compute reconciliation reports
        reconciliation_reports = simulate_reconciliation(state, actions, next_state)
        
        # 5. FEEDBACK: Agents receive feedback and compute rewards
        for agent_id, agent in agents.items():
            reward = agent.receive_feedback(reconciliation_reports[agent_id])
            episode_rewards[agent_id].append(reward)
        
        state = next_state
    
    # Mark episode end for learning wrappers
    for agent in agents.values():
        if isinstance(agent, LearningWrapper):
            agent.mark_episode_end()
    
    # Compute episode statistics
    stats = {
        agent_id: {
            'total_reward': sum(rewards),
            'avg_reward': np.mean(rewards),
            'num_steps': len(rewards)
        }
        for agent_id, rewards in episode_rewards.items()
    }
    
    return stats


def main():
    """
    Main training loop for HMARL system.
    """
    print("="*60)
    print("HMARL TRAINING - Store Agents with PPO")
    print("="*60)
    
    # Configuration
    num_episodes = 10
    num_days_per_episode = 30
    update_frequency = 1  # Update after each episode
    
    # Initialize environment
    sim_config = {
        'SIMULATION_DAYS': num_days_per_episode,
        'WARMUP_DAYS': 0,  # No warmup for training
        'RANDOM_SEED': 42,
        'SKU_CONFIG': config.SKU_CONFIG,
        'STORE_CONFIG': config.STORE_CONFIG,
        'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
        'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
    }
    
    env = DigitalTwin(sim_config)
    
    # Initialize PPO trainer for store agents
    ppo_trainer = PPOTrainer(
        obs_dim=StoreAgent.OBSERVATION_DIM,
        action_dim=StoreAgent.ACTION_DIM,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        shared_policy=True  # Share policy across store agents
    )
    
    # Create agents (learning enabled for stores)
    agents = create_agents(ppo_trainer=ppo_trainer, learning_enabled=True)
    
    print(f"\nAgents initialized:")
    for agent_id, agent in agents.items():
        print(f"  - {agent_id}: {agent}")
    
    print(f"\nTraining configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Days per episode: {num_days_per_episode}")
    print(f"  Update frequency: every {update_frequency} episode(s)")
    print()
    
    # Training loop
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # Run episode
        stats = train_episode(env, agents, num_days=num_days_per_episode)
        
        # Print episode statistics
        for agent_id, agent_stats in stats.items():
            print(f"{agent_id}: Total Reward = {agent_stats['total_reward']:.2f}, "
                  f"Avg Reward = {agent_stats['avg_reward']:.2f}")
        
        # Update learners
        if (episode + 1) % update_frequency == 0:
            print("\nUpdating learners...")
            
            # Collect experiences from all learning wrappers
            all_experiences = []
            for agent in agents.values():
                if isinstance(agent, LearningWrapper) and agent.learning_enabled:
                    all_experiences.extend(agent.get_experiences())
            
            # Update PPO trainer with pooled experiences
            if all_experiences:
                ppo_trainer.update(all_experiences, num_epochs=4)
                print(f"  Updated with {len(all_experiences)} experiences")
                
                # Clear experience buffers
                for agent in agents.values():
                    if isinstance(agent, LearningWrapper):
                        agent.clear_experiences()
            
            # Print training statistics
            trainer_stats = ppo_trainer.get_stats()
            print(f"  PPO Stats: {trainer_stats}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # Save trained model
    checkpoint_path = 'checkpoints/ppo_store_agents.pt'
    os.makedirs('checkpoints', exist_ok=True)
    ppo_trainer.save_checkpoint(checkpoint_path)
    print(f"\nModel saved to: {checkpoint_path}")
    
    # Final agent statistics
    print("\nFinal Agent Statistics:")
    for agent_id, agent in agents.items():
        if isinstance(agent, LearningWrapper):
            print(f"\n{agent_id}:")
            for key, value in agent.get_stats().items():
                print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
