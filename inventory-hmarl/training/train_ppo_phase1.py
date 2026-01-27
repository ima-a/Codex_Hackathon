"""
train_ppo_phase1.py

Phase-1 PPO Training for HMARL System.

Trains ONLY the 2 Store agents using:
- Shared PPO policy (parameter sharing)
- Centralized Training, Decentralized Execution (CTDE)
- Reconciliation-driven rewards

Warehouse and Supplier agents remain rule-based.

Requirements:
- Environment validation must pass first
- Stable-Baselines3 for PPO
- Generates training plots and comparisons
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.hmarl_env import HMARLEnvironment, SingleAgentWrapper
import config.simulation_config as config

# Stable-Baselines3 imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    print("WARNING: Stable-Baselines3 not available. Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False


class TrainingMetricsCallback(BaseCallback):
    """
    Custom callback to log training metrics.
    
    Tracks:
    - Episode rewards
    - Service levels
    - Stockouts
    - Holding costs
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_service_levels = []
        self.episode_stockouts = []
        self.episode_holding_costs = []
        self.episode_lengths = []
        
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        """Called at each step."""
        # Get info from environment
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # Track episode metrics
            self.current_episode_reward += self.locals.get('rewards', [0])[0]
            self.current_episode_length += 1
            
            # Check if episode is done
            if self.locals.get('dones', [False])[0]:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                
                # Extract reconciliation metrics if available
                if 'reconciliation' in info:
                    recon = info['reconciliation']
                    
                    # Average across store agents
                    service_levels = []
                    stockouts = []
                    holding_costs = []
                    
                    for agent_id, metrics in recon.items():
                        if 'store' in agent_id:
                            service_levels.append(metrics.get('service_level', 0))
                            stockouts.append(metrics.get('lost_sales', 0))
                            holding_costs.append(metrics.get('holding_cost', 0))
                    
                    if service_levels:
                        self.episode_service_levels.append(np.mean(service_levels))
                    if stockouts:
                        self.episode_stockouts.append(np.sum(stockouts))
                    if holding_costs:
                        self.episode_holding_costs.append(np.sum(holding_costs))
                
                # Reset episode tracking
                self.current_episode_reward = 0
                self.current_episode_length = 0
        
        return True
    
    def get_metrics(self) -> Dict:
        """Get collected metrics."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_service_levels': self.episode_service_levels,
            'episode_stockouts': self.episode_stockouts,
            'episode_holding_costs': self.episode_holding_costs,
            'episode_lengths': self.episode_lengths
        }


class PPOTrainerPhase1:
    """
    Phase-1 PPO trainer for Store agents.
    
    Features:
    - Parameter sharing across store agents
    - CTDE architecture
    - Reconciliation-driven rewards
    - Training visualization
    - Baseline comparison
    """
    
    def __init__(
        self,
        env_config: Dict,
        output_dir: str = 'training_outputs',
        total_timesteps: int = 10000,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        verbose: int = 1
    ):
        """
        Initialize Phase-1 PPO trainer.
        
        Args:
            env_config: Environment configuration
            output_dir: Directory for outputs
            total_timesteps: Total training timesteps
            learning_rate: PPO learning rate
            n_steps: Steps per rollout
            batch_size: Minibatch size
            n_epochs: Optimization epochs per rollout
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clip range
            ent_coef: Entropy coefficient
            verbose: Verbosity level
        """
        self.env_config = env_config
        self.output_dir = output_dir
        self.total_timesteps = total_timesteps
        
        # PPO hyperparameters
        self.ppo_params = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_range': clip_range,
            'ent_coef': ent_coef,
            'verbose': verbose
        }
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training artifacts
        self.model = None
        self.env = None
        self.metrics_callback = None
        self.training_start_time = None
    
    def create_environment(self, agent_id: str = 'store_store_1') -> SingleAgentWrapper:
        """
        Create single-agent wrapper for one store agent.
        
        For parameter sharing, we train on one agent's view but the
        policy is shared across both store agents.
        
        Args:
            agent_id: Agent to expose (default: first store)
        
        Returns:
            SingleAgentWrapper environment
        """
        # Create multi-agent environment
        multi_env = HMARLEnvironment(
            config=self.env_config,
            max_steps=30
        )
        
        # Wrap for single-agent training
        single_env = SingleAgentWrapper(multi_env, agent_id=agent_id)
        
        return single_env
    
    def train(self):
        """
        Run Phase-1 PPO training.
        
        Trains store agents with shared policy.
        """
        if not SB3_AVAILABLE:
            raise ImportError("Stable-Baselines3 required. Install with: pip install stable-baselines3")
        
        print("="*60)
        print("PHASE-1 PPO TRAINING")
        print("="*60)
        print(f"\nTraining Configuration:")
        print(f"  Total timesteps: {self.total_timesteps:,}")
        print(f"  Learning rate: {self.ppo_params['learning_rate']}")
        print(f"  Batch size: {self.ppo_params['batch_size']}")
        print(f"  Epochs per rollout: {self.ppo_params['n_epochs']}")
        print(f"  Gamma: {self.ppo_params['gamma']}")
        print(f"  GAE lambda: {self.ppo_params['gae_lambda']}")
        print(f"  Clip range: {self.ppo_params['clip_range']}")
        print(f"  Entropy coef: {self.ppo_params['ent_coef']}")
        print(f"\nOutput directory: {self.output_dir}")
        
        # Create environment
        print(f"\nCreating environment...")
        self.env = DummyVecEnv([lambda: self.create_environment()])
        
        # Create metrics callback
        self.metrics_callback = TrainingMetricsCallback(verbose=1)
        
        # Create PPO model
        print(f"\nInitializing PPO model...")
        self.model = PPO(
            "MlpPolicy",
            self.env,
            **self.ppo_params
        )
        
        print(f"\nModel architecture:")
        print(self.model.policy)
        
        # Start training
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}\n")
        
        self.training_start_time = datetime.now()
        
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=self.metrics_callback
        )
        
        training_duration = (datetime.now() - self.training_start_time).total_seconds()
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"{'='*60}")
        print(f"Duration: {training_duration:.1f} seconds")
        print(f"Episodes: {len(self.metrics_callback.episode_rewards)}")
        
        # Save model
        model_path = os.path.join(self.output_dir, 'ppo_store_agents_phase1.zip')
        self.model.save(model_path)
        print(f"\nModel saved to: {model_path}")
        
        # Save training metrics
        self.save_metrics()
        
        # Generate plots
        self.generate_plots()
    
    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics = self.metrics_callback.get_metrics()
        
        # Convert numpy arrays to lists for JSON
        metrics_json = {
            key: [float(v) for v in value]
            for key, value in metrics.items()
        }
        
        metrics_path = os.path.join(self.output_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        print(f"Metrics saved to: {metrics_path}")
    
    def generate_plots(self):
        """Generate training visualization plots."""
        print(f"\nGenerating training plots...")
        
        metrics = self.metrics_callback.get_metrics()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase-1 PPO Training Results - Store Agents', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode Rewards
        if metrics['episode_rewards']:
            ax = axes[0, 0]
            episodes = range(1, len(metrics['episode_rewards']) + 1)
            ax.plot(episodes, metrics['episode_rewards'], label='Episode Reward', alpha=0.6)
            
            # Moving average
            window = min(10, len(metrics['episode_rewards']) // 5)
            if window > 1:
                moving_avg = np.convolve(metrics['episode_rewards'], np.ones(window)/window, mode='valid')
                ax.plot(range(window, len(metrics['episode_rewards']) + 1), moving_avg, 
                       label=f'Moving Avg ({window} eps)', linewidth=2, color='red')
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.set_title('Episode Rewards Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Service Level
        if metrics['episode_service_levels']:
            ax = axes[0, 1]
            episodes = range(1, len(metrics['episode_service_levels']) + 1)
            ax.plot(episodes, metrics['episode_service_levels'], label='Service Level', color='green', alpha=0.6)
            
            # Target line
            ax.axhline(y=0.95, color='red', linestyle='--', label='Target (95%)', linewidth=2)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Service Level')
            ax.set_title('Service Level Over Time')
            ax.set_ylim([0, 1.05])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Stockouts
        if metrics['episode_stockouts']:
            ax = axes[1, 0]
            episodes = range(1, len(metrics['episode_stockouts']) + 1)
            ax.plot(episodes, metrics['episode_stockouts'], label='Stockouts', color='orange', alpha=0.6)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Stockouts')
            ax.set_title('Stockouts Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Holding Costs
        if metrics['episode_holding_costs']:
            ax = axes[1, 1]
            episodes = range(1, len(metrics['episode_holding_costs']) + 1)
            ax.plot(episodes, metrics['episode_holding_costs'], label='Holding Cost', color='purple', alpha=0.6)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Holding Cost ($)')
            ax.set_title('Holding Costs Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'training_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to: {plot_path}")
        
        plt.close()
    
    def evaluate(self, n_episodes: int = 10) -> Dict:
        """
        Evaluate trained policy.
        
        Args:
            n_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION ({n_episodes} episodes)")
        print(f"{'='*60}")
        
        eval_rewards = []
        eval_service_levels = []
        eval_stockouts = []
        eval_holding_costs = []
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward[0]
            
            eval_rewards.append(episode_reward)
            
            # Extract metrics from last info
            if len(info) > 0 and 'reconciliation' in info[0]:
                recon = info[0]['reconciliation']
                
                service_levels = []
                stockouts = []
                holding_costs = []
                
                for agent_id, metrics in recon.items():
                    if 'store' in agent_id:
                        service_levels.append(metrics.get('service_level', 0))
                        stockouts.append(metrics.get('lost_sales', 0))
                        holding_costs.append(metrics.get('holding_cost', 0))
                
                if service_levels:
                    eval_service_levels.append(np.mean(service_levels))
                if stockouts:
                    eval_stockouts.append(np.sum(stockouts))
                if holding_costs:
                    eval_holding_costs.append(np.sum(holding_costs))
        
        # Print results
        print(f"\nEvaluation Results:")
        print(f"  Avg Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        if eval_service_levels:
            print(f"  Avg Service Level: {np.mean(eval_service_levels):.3f}")
        if eval_stockouts:
            print(f"  Avg Stockouts: {np.mean(eval_stockouts):.1f}")
        if eval_holding_costs:
            print(f"  Avg Holding Cost: ${np.mean(eval_holding_costs):.2f}")
        
        return {
            'rewards': eval_rewards,
            'service_levels': eval_service_levels,
            'stockouts': eval_stockouts,
            'holding_costs': eval_holding_costs
        }


def main():
    """Main training pipeline."""
    # Environment configuration
    env_config = {
        'SIMULATION_DAYS': 30,
        'WARMUP_DAYS': 0,
        'RANDOM_SEED': 42,
        'SKU_CONFIG': config.SKU_CONFIG,
        'STORE_CONFIG': config.STORE_CONFIG,
        'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
        'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
    }
    
    # Create trainer
    trainer = PPOTrainerPhase1(
        env_config=env_config,
        output_dir='training_outputs/phase1',
        total_timesteps=10000,  # Hackathon-scale
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        verbose=1
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    trainer.evaluate(n_episodes=10)
    
    print(f"\n{'='*60}")
    print("PHASE-1 TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Review training plots in: {trainer.output_dir}")
    print(f"  2. Compare against baseline policies")
    print(f"  3. Proceed to Phase-2 (warehouse agent training)")


if __name__ == '__main__':
    main()
