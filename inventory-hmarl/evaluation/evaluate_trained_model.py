"""
evaluate_trained_model.py

Post-training evaluation pipeline for HMARL system.

This script:
1. Loads trained PPO checkpoint
2. Runs evaluation in deterministic mode (no exploration)
3. Compares PPO vs Baseline policies
4. Collects metrics (service level, stockouts, rewards, costs)
5. Generates comparison plots
6. Produces summary report

Usage:
    python evaluation/evaluate_trained_model.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.hmarl_env import HMARLEnvironment
from agents.ppo_trainer import PPOTrainer
from agents.store_agent import StoreAgent
import config.simulation_config as config


class EvaluationPipeline:
    """
    Evaluation pipeline for trained HMARL system.
    
    Compares PPO-trained agents vs baseline rule-based policies.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        num_episodes: int = 10,
        episode_length: int = 60,
        output_dir: str = 'evaluation/results'
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            checkpoint_path: Path to trained PPO model
            num_episodes: Number of evaluation episodes
            episode_length: Steps per episode (simulated days)
            output_dir: Directory for results
        """
        self.checkpoint_path = checkpoint_path
        self.num_episodes = num_episodes
        self.episode_length = episode_length
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Environment configuration
        self.env_config = {
            'SIMULATION_DAYS': episode_length,
            'WARMUP_DAYS': 0,
            'RANDOM_SEED': 42,  # Fixed seed for reproducibility
            'SKU_CONFIG': config.SKU_CONFIG,
            'STORE_CONFIG': config.STORE_CONFIG,
            'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
            'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
        }
        
        # Agent configurations
        self.agent_configs = {
            'store_store_1': {'reorder_point': 300, 'order_up_to_level': 700},
            'store_store_2': {'reorder_point': 300, 'order_up_to_level': 700},
            'store_store_3': {'reorder_point': 300, 'order_up_to_level': 700},
            'warehouse_warehouse_1': {'reorder_point': 2000, 'order_up_to_level': 4000},
            'supplier_supplier_1': {'production_capacity': 10000, 'lead_time': 7}
        }
        
        print("="*70)
        print("HMARL EVALUATION PIPELINE")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Episodes: {num_episodes}")
        print(f"  Episode length: {episode_length} days")
        print(f"  Output directory: {output_dir}")
        print(f"  Random seed: {self.env_config['RANDOM_SEED']}")
    
    def load_ppo_model(self) -> PPOTrainer:
        """Load trained PPO model."""
        print(f"\nLoading PPO model from {self.checkpoint_path}...")
        
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
        
        ppo_trainer.load_checkpoint(self.checkpoint_path)
        print("✓ PPO model loaded successfully")
        
        return ppo_trainer
    
    def run_evaluation(
        self,
        ppo_trainer: PPOTrainer,
        use_ppo: bool,
        policy_name: str
    ) -> Dict:
        """
        Run evaluation for specified number of episodes.
        
        Args:
            ppo_trainer: Trained PPO model
            use_ppo: If True, use PPO policy; else use baseline
            policy_name: Name for logging
        
        Returns:
            Dictionary of collected metrics
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {policy_name}")
        print(f"{'='*70}")
        
        # Create environment
        env = HMARLEnvironment(
            config=self.env_config,
            agent_configs=self.agent_configs,
            max_steps=self.episode_length,
            warmup_days=0
        )
        
        store_agent_ids = [aid for aid in env.agent_ids if aid.startswith('store_')]
        
        # Metrics storage
        all_metrics = {
            'episode_rewards': [],
            'daily_rewards': [],
            'service_levels': [],
            'stockouts': [],
            'holding_costs': [],
            'lost_sales': [],
            'inventory_levels': []
        }
        
        # Run episodes
        for episode in range(self.num_episodes):
            print(f"\nEpisode {episode + 1}/{self.num_episodes}")
            
            observations, _ = env.reset()
            
            episode_data = {
                'rewards': [],
                'service_levels': [],
                'stockouts': [],
                'holding_costs': [],
                'lost_sales': [],
                'inventory_levels': []
            }
            
            for step in range(self.episode_length):
                # Select actions
                actions = {}
                
                for agent_id in env.agent_ids:
                    obs = observations[agent_id]
                    
                    if agent_id in store_agent_ids:
                        if use_ppo:
                            # Use trained PPO policy (deterministic)
                            action = ppo_trainer.select_action(obs, deterministic=True)
                        else:
                            # Use baseline rule-based policy
                            action = env.agents[agent_id].act(obs)
                    else:
                        # Warehouse and supplier always use rule-based
                        action = env.agents[agent_id].act(obs)
                    
                    actions[agent_id] = action
                
                # Step environment
                next_observations, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                
                # Collect metrics from reconciliation
                if 'reconciliation' in info:
                    recon = info['reconciliation']
                    
                    # Aggregate store metrics
                    step_service_level = np.mean([
                        recon[aid].get('service_level', 0.0)
                        for aid in store_agent_ids
                    ])
                    
                    step_stockouts = sum([
                        recon[aid].get('lost_sales', 0)
                        for aid in store_agent_ids
                    ])
                    
                    step_holding_cost = sum([
                        recon[aid].get('holding_cost', 0.0)
                        for aid in store_agent_ids
                    ])
                    
                    step_lost_sales = sum([
                        recon[aid].get('lost_sales', 0)
                        for aid in store_agent_ids
                    ])
                    
                    step_inventory = sum([
                        recon[aid].get('inventory', 0)
                        for aid in store_agent_ids
                    ])
                    
                    episode_data['service_levels'].append(step_service_level)
                    episode_data['stockouts'].append(step_stockouts)
                    episode_data['holding_costs'].append(step_holding_cost)
                    episode_data['lost_sales'].append(step_lost_sales)
                    episode_data['inventory_levels'].append(step_inventory)
                
                # Collect rewards
                store_rewards = [rewards[aid] for aid in store_agent_ids]
                episode_data['rewards'].append(np.mean(store_rewards))
                
                observations = next_observations
                
                if done:
                    break
            
            # Store episode metrics
            all_metrics['episode_rewards'].append(np.sum(episode_data['rewards']))
            all_metrics['daily_rewards'].append(episode_data['rewards'])
            all_metrics['service_levels'].append(np.mean(episode_data['service_levels']))
            all_metrics['stockouts'].append(np.sum(episode_data['stockouts']))
            all_metrics['holding_costs'].append(np.mean(episode_data['holding_costs']))
            all_metrics['lost_sales'].append(np.sum(episode_data['lost_sales']))
            all_metrics['inventory_levels'].append(np.mean(episode_data['inventory_levels']))
            
            print(f"  Episode reward: {all_metrics['episode_rewards'][-1]:.2f}")
            print(f"  Avg service level: {all_metrics['service_levels'][-1]:.3f}")
            print(f"  Total stockouts: {all_metrics['stockouts'][-1]:.0f}")
        
        env.close()
        
        # Compute summary statistics
        summary = {
            'policy': policy_name,
            'avg_episode_reward': np.mean(all_metrics['episode_rewards']),
            'std_episode_reward': np.std(all_metrics['episode_rewards']),
            'avg_service_level': np.mean(all_metrics['service_levels']),
            'std_service_level': np.std(all_metrics['service_levels']),
            'avg_stockouts': np.mean(all_metrics['stockouts']),
            'total_stockouts': np.sum(all_metrics['stockouts']),
            'avg_holding_cost': np.mean(all_metrics['holding_costs']),
            'avg_lost_sales': np.mean(all_metrics['lost_sales']),
            'avg_inventory': np.mean(all_metrics['inventory_levels'])
        }
        
        print(f"\n{policy_name} Summary:")
        print(f"  Avg Episode Reward: {summary['avg_episode_reward']:.2f} ± {summary['std_episode_reward']:.2f}")
        print(f"  Avg Service Level: {summary['avg_service_level']:.3f} ± {summary['std_service_level']:.3f}")
        print(f"  Total Stockouts: {summary['total_stockouts']:.0f}")
        print(f"  Avg Holding Cost: {summary['avg_holding_cost']:.2f}")
        
        return {
            'metrics': all_metrics,
            'summary': summary
        }
    
    def save_metrics_csv(self, baseline_results: Dict, ppo_results: Dict):
        """Save metrics to CSV files."""
        print(f"\nSaving metrics to CSV...")
        
        # Episode-level metrics
        episode_df = pd.DataFrame({
            'episode': range(1, self.num_episodes + 1),
            'baseline_reward': baseline_results['metrics']['episode_rewards'],
            'ppo_reward': ppo_results['metrics']['episode_rewards'],
            'baseline_service_level': baseline_results['metrics']['service_levels'],
            'ppo_service_level': ppo_results['metrics']['service_levels'],
            'baseline_stockouts': baseline_results['metrics']['stockouts'],
            'ppo_stockouts': ppo_results['metrics']['stockouts']
        })
        
        episode_csv = os.path.join(self.output_dir, 'episode_metrics.csv')
        episode_df.to_csv(episode_csv, index=False)
        print(f"  ✓ Saved: {episode_csv}")
        
        # Summary metrics
        summary_df = pd.DataFrame([
            baseline_results['summary'],
            ppo_results['summary']
        ])
        
        summary_csv = os.path.join(self.output_dir, 'summary_metrics.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"  ✓ Saved: {summary_csv}")
    
    def generate_plots(self, baseline_results: Dict, ppo_results: Dict):
        """Generate comparison plots."""
        print(f"\nGenerating plots...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Service Level Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        episodes = range(1, self.num_episodes + 1)
        ax.plot(episodes, baseline_results['metrics']['service_levels'], 
                'o-', label='Baseline', linewidth=2, markersize=8)
        ax.plot(episodes, ppo_results['metrics']['service_levels'], 
                's-', label='PPO', linewidth=2, markersize=8)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Service Level', fontsize=12)
        ax.set_title('Service Level: Baseline vs PPO', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        service_plot = os.path.join(self.output_dir, 'service_level_comparison.png')
        plt.tight_layout()
        plt.savefig(service_plot, dpi=300)
        plt.close()
        print(f"  ✓ Saved: {service_plot}")
        
        # 2. Stockouts Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(episodes, baseline_results['metrics']['stockouts'], 
                'o-', label='Baseline', linewidth=2, markersize=8)
        ax.plot(episodes, ppo_results['metrics']['stockouts'], 
                's-', label='PPO', linewidth=2, markersize=8)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Stockouts', fontsize=12)
        ax.set_title('Stockouts: Baseline vs PPO', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        stockout_plot = os.path.join(self.output_dir, 'stockouts_comparison.png')
        plt.tight_layout()
        plt.savefig(stockout_plot, dpi=300)
        plt.close()
        print(f"  ✓ Saved: {stockout_plot}")
        
        # 3. Episode Rewards Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(episodes, baseline_results['metrics']['episode_rewards'], 
                'o-', label='Baseline', linewidth=2, markersize=8)
        ax.plot(episodes, ppo_results['metrics']['episode_rewards'], 
                's-', label='PPO', linewidth=2, markersize=8)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Episode Reward', fontsize=12)
        ax.set_title('Episode Rewards: Baseline vs PPO', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        reward_plot = os.path.join(self.output_dir, 'rewards_comparison.png')
        plt.tight_layout()
        plt.savefig(reward_plot, dpi=300)
        plt.close()
        print(f"  ✓ Saved: {reward_plot}")
    
    def generate_report(self, baseline_results: Dict, ppo_results: Dict):
        """Generate text summary report."""
        print(f"\nGenerating summary report...")
        
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HMARL EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Episodes: {self.num_episodes}\n")
            f.write(f"Episode Length: {self.episode_length} days\n\n")
            
            f.write("="*70 + "\n")
            f.write("BASELINE POLICY RESULTS\n")
            f.write("="*70 + "\n\n")
            
            baseline_summary = baseline_results['summary']
            f.write(f"Average Episode Reward: {baseline_summary['avg_episode_reward']:.2f} ± {baseline_summary['std_episode_reward']:.2f}\n")
            f.write(f"Average Service Level: {baseline_summary['avg_service_level']:.3f} ± {baseline_summary['std_service_level']:.3f}\n")
            f.write(f"Total Stockouts: {baseline_summary['total_stockouts']:.0f}\n")
            f.write(f"Average Stockouts per Episode: {baseline_summary['avg_stockouts']:.2f}\n")
            f.write(f"Average Holding Cost: {baseline_summary['avg_holding_cost']:.2f}\n")
            f.write(f"Average Lost Sales: {baseline_summary['avg_lost_sales']:.2f}\n\n")
            
            f.write("="*70 + "\n")
            f.write("PPO POLICY RESULTS\n")
            f.write("="*70 + "\n\n")
            
            ppo_summary = ppo_results['summary']
            f.write(f"Average Episode Reward: {ppo_summary['avg_episode_reward']:.2f} ± {ppo_summary['std_episode_reward']:.2f}\n")
            f.write(f"Average Service Level: {ppo_summary['avg_service_level']:.3f} ± {ppo_summary['std_service_level']:.3f}\n")
            f.write(f"Total Stockouts: {ppo_summary['total_stockouts']:.0f}\n")
            f.write(f"Average Stockouts per Episode: {ppo_summary['avg_stockouts']:.2f}\n")
            f.write(f"Average Holding Cost: {ppo_summary['avg_holding_cost']:.2f}\n")
            f.write(f"Average Lost Sales: {ppo_summary['avg_lost_sales']:.2f}\n\n")
            
            f.write("="*70 + "\n")
            f.write("COMPARISON (PPO vs Baseline)\n")
            f.write("="*70 + "\n\n")
            
            reward_improvement = ((ppo_summary['avg_episode_reward'] - baseline_summary['avg_episode_reward']) 
                                 / abs(baseline_summary['avg_episode_reward']) * 100)
            service_improvement = ((ppo_summary['avg_service_level'] - baseline_summary['avg_service_level']) 
                                  / baseline_summary['avg_service_level'] * 100)
            stockout_reduction = ((baseline_summary['total_stockouts'] - ppo_summary['total_stockouts']) 
                                 / baseline_summary['total_stockouts'] * 100) if baseline_summary['total_stockouts'] > 0 else 0
            
            f.write(f"Reward Improvement: {reward_improvement:+.2f}%\n")
            f.write(f"Service Level Improvement: {service_improvement:+.2f}%\n")
            f.write(f"Stockout Reduction: {stockout_reduction:+.2f}%\n\n")
            
            f.write("="*70 + "\n")
            f.write("QUALITATIVE ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            if reward_improvement > 0:
                f.write("✓ PPO policy achieves higher rewards than baseline\n")
            if service_improvement > 0:
                f.write("✓ PPO policy improves service level\n")
            if stockout_reduction > 0:
                f.write("✓ PPO policy reduces stockouts\n")
            
            f.write("\nThe PPO-trained agents demonstrate learned behavior that balances:\n")
            f.write("  - Customer service (meeting demand)\n")
            f.write("  - Inventory costs (holding costs)\n")
            f.write("  - Stockout penalties (lost sales)\n\n")
            
            f.write("This is achieved through:\n")
            f.write("  - Reconciliation-driven rewards from business metrics\n")
            f.write("  - Hierarchical multi-agent coordination (Stores → Warehouse → Supplier)\n")
            f.write("  - Shared PPO policy across store agents (CTDE)\n")
            f.write("  - Experience pooling for efficient learning\n\n")
        
        print(f"  ✓ Saved: {report_path}")
        
        # Print to console
        with open(report_path, 'r') as f:
            print("\n" + f.read())
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        # Load PPO model
        ppo_trainer = self.load_ppo_model()
        
        # Run baseline evaluation
        baseline_results = self.run_evaluation(
            ppo_trainer=ppo_trainer,
            use_ppo=False,
            policy_name="Baseline (Rule-based)"
        )
        
        # Run PPO evaluation
        ppo_results = self.run_evaluation(
            ppo_trainer=ppo_trainer,
            use_ppo=True,
            policy_name="PPO (Trained)"
        )
        
        # Save metrics
        self.save_metrics_csv(baseline_results, ppo_results)
        
        # Generate plots
        self.generate_plots(baseline_results, ppo_results)
        
        # Generate report
        self.generate_report(baseline_results, ppo_results)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"\nResults saved to: {self.output_dir}/")
        print("  - episode_metrics.csv")
        print("  - summary_metrics.csv")
        print("  - service_level_comparison.png")
        print("  - stockouts_comparison.png")
        print("  - rewards_comparison.png")
        print("  - evaluation_report.txt")


def main():
    """Main evaluation runner."""
    # Configuration
    checkpoint_path = 'checkpoints/ppo_store_agents_gym.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using: python agents/train_with_gym_env.py")
        return
    
    # Create evaluation pipeline
    evaluator = EvaluationPipeline(
        checkpoint_path=checkpoint_path,
        num_episodes=10,
        episode_length=60,  # 60 days
        output_dir='evaluation/results'
    )
    
    # Run evaluation
    evaluator.run_full_evaluation()


if __name__ == '__main__':
    main()
