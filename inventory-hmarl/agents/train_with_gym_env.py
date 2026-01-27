"""
train_with_gym_env.py

Training script using Gym-compatible HMARL environment.

Demonstrates:
- Using HMARLEnvironment with standard Gym interface
- Multi-agent PPO training
- Centralized training with experience pooling
- Integration with reconciliation-driven rewards
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.hmarl_env import HMARLEnvironment, SingleAgentWrapper
from agents.ppo_trainer import PPOTrainer
from agents.store_agent import StoreAgent
import config.simulation_config as config


def train_multi_agent_ppo(num_episodes=20, num_steps=30, verbose=True):
    """
    Train store agents using multi-agent PPO.
    
    Args:
        num_episodes: Number of training episodes
        num_steps: Steps per episode
        verbose: Print progress
    """
    if verbose:
        print("="*60)
        print("MULTI-AGENT PPO TRAINING WITH GYM ENVIRONMENT")
        print("="*60)
    
    # Environment configuration
    env_config = {
        'SIMULATION_DAYS': num_steps,
        'WARMUP_DAYS': 0,
        'RANDOM_SEED': 42,
        'SKU_CONFIG': config.SKU_CONFIG,
        'STORE_CONFIG': config.STORE_CONFIG,
        'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
        'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
    }
    
    # Agent configurations
    agent_configs = {
        'store_store_1': {'reorder_point': 300, 'order_up_to_level': 700},
        'store_store_2': {'reorder_point': 300, 'order_up_to_level': 700},
        'warehouse_warehouse_1': {'reorder_point': 2000, 'order_up_to_level': 4000},
        'supplier_supplier_1': {'production_capacity': 10000, 'lead_time': 7}
    }
    
    # Create environment
    env = HMARLEnvironment(
        config=env_config,
        agent_configs=agent_configs,
        max_steps=num_steps,
        warmup_days=0
    )
    
    if verbose:
        print(f"\nEnvironment created:")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Agents: {env.agent_ids}")
    
    # Create PPO trainer for store agents (shared policy)
    store_agent_ids = [aid for aid in env.agent_ids if aid.startswith('store_')]
    
    ppo_trainer = PPOTrainer(
        obs_dim=StoreAgent.OBSERVATION_DIM,
        action_dim=StoreAgent.ACTION_DIM,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        shared_policy=True
    )
    
    if verbose:
        print(f"\nPPO Trainer initialized:")
        print(f"  Store agents (learning): {store_agent_ids}")
        print(f"  Shared policy: Yes")
        print(f"  Observation dim: {StoreAgent.OBSERVATION_DIM}")
        print(f"  Action dim: {StoreAgent.ACTION_DIM}")
    
    # Training loop
    all_episode_stats = []
    
    for episode in range(num_episodes):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*60}")
        
        # Reset environment
        observations = env.reset()
        
        # Experience buffer for this episode
        episode_experiences = []
        episode_rewards = {aid: [] for aid in env.agent_ids}
        
        # Run episode
        for step in range(num_steps):
            # Select actions
            actions = {}
            
            for agent_id in env.agent_ids:
                obs = observations[agent_id]
                
                if agent_id in store_agent_ids:
                    # Use PPO policy for store agents
                    action = ppo_trainer.select_action(obs, deterministic=False)
                else:
                    # Use rule-based policy for other agents
                    action = env.agents[agent_id].act(obs)
                
                actions[agent_id] = action
            
            # Step environment
            next_observations, rewards, done, info = env.step(actions)
            
            # Store experiences for store agents
            for agent_id in store_agent_ids:
                experience = {
                    'observation': observations[agent_id],
                    'action': actions[agent_id],
                    'reward': rewards[agent_id],
                    'done': done
                }
                episode_experiences.append(experience)
            
            # Track rewards
            for agent_id in env.agent_ids:
                episode_rewards[agent_id].append(rewards[agent_id])
            
            # Update observations
            observations = next_observations
            
            if done:
                break
        
        # Episode statistics
        episode_stats = env.get_episode_stats()
        all_episode_stats.append(episode_stats)
        
        if verbose:
            print(f"\nEpisode {episode + 1} Results:")
            for agent_id, stats in episode_stats.items():
                print(f"  {agent_id}:")
                print(f"    Total Reward: {stats['total_reward']:.2f}")
                print(f"    Avg Reward: {stats['avg_reward']:.2f}")
        
        # Update PPO trainer
        if episode_experiences:
            ppo_trainer.update(episode_experiences, num_epochs=4)
            
            if verbose:
                print(f"\nPPO Update:")
                print(f"  Experiences: {len(episode_experiences)}")
                trainer_stats = ppo_trainer.get_stats()
                print(f"  Avg Policy Loss: {trainer_stats['avg_policy_loss']:.4f}")
                print(f"  Avg Value Loss: {trainer_stats['avg_value_loss']:.4f}")
    
    # Save trained model
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'ppo_store_agents_gym.pt')
    ppo_trainer.save_checkpoint(checkpoint_path)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"\nModel saved to: {checkpoint_path}")
        
        # Print final statistics
        print(f"\nFinal Training Statistics:")
        print(f"  Total episodes: {num_episodes}")
        print(f"  Steps per episode: {num_steps}")
        
        # Average rewards across all episodes
        print(f"\nAverage Performance (all episodes):")
        for agent_id in env.agent_ids:
            avg_total_reward = np.mean([stats[agent_id]['total_reward'] for stats in all_episode_stats])
            avg_avg_reward = np.mean([stats[agent_id]['avg_reward'] for stats in all_episode_stats])
            print(f"  {agent_id}:")
            print(f"    Avg Total Reward: {avg_total_reward:.2f}")
            print(f"    Avg Step Reward: {avg_avg_reward:.2f}")
    
    env.close()
    
    return ppo_trainer, all_episode_stats


def evaluate_trained_policy(ppo_trainer, num_episodes=5, num_steps=30):
    """
    Evaluate trained policy.
    
    Args:
        ppo_trainer: Trained PPO trainer
        num_episodes: Number of evaluation episodes
        num_steps: Steps per episode
    """
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    
    # Create evaluation environment
    env_config = {
        'SIMULATION_DAYS': num_steps,
        'WARMUP_DAYS': 0,
        'RANDOM_SEED': 123,  # Different seed for evaluation
        'SKU_CONFIG': config.SKU_CONFIG,
        'STORE_CONFIG': config.STORE_CONFIG,
        'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
        'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
    }
    
    env = HMARLEnvironment(config=env_config, max_steps=num_steps)
    
    store_agent_ids = [aid for aid in env.agent_ids if aid.startswith('store_')]
    
    eval_stats = []
    
    for episode in range(num_episodes):
        observations = env.reset()
        episode_rewards = {aid: [] for aid in env.agent_ids}
        
        for step in range(num_steps):
            actions = {}
            
            for agent_id in env.agent_ids:
                obs = observations[agent_id]
                
                if agent_id in store_agent_ids:
                    # Use trained policy (deterministic)
                    action = ppo_trainer.select_action(obs, deterministic=True)
                else:
                    action = env.agents[agent_id].act(obs)
                
                actions[agent_id] = action
            
            next_observations, rewards, done, info = env.step(actions)
            
            for agent_id in env.agent_ids:
                episode_rewards[agent_id].append(rewards[agent_id])
            
            observations = next_observations
            
            if done:
                break
        
        episode_stats = env.get_episode_stats()
        eval_stats.append(episode_stats)
        
        print(f"\nEvaluation Episode {episode + 1}:")
        for agent_id in store_agent_ids:
            print(f"  {agent_id}: Total Reward = {episode_stats[agent_id]['total_reward']:.2f}")
    
    # Average evaluation performance
    print(f"\nAverage Evaluation Performance:")
    for agent_id in store_agent_ids:
        avg_total = np.mean([stats[agent_id]['total_reward'] for stats in eval_stats])
        print(f"  {agent_id}: Avg Total Reward = {avg_total:.2f}")
    
    env.close()
    
    return eval_stats


def main():
    """Main training and evaluation."""
    # Train
    ppo_trainer, training_stats = train_multi_agent_ppo(
        num_episodes=20,
        num_steps=30,
        verbose=True
    )
    
    # Evaluate
    eval_stats = evaluate_trained_policy(
        ppo_trainer,
        num_episodes=5,
        num_steps=30
    )
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
