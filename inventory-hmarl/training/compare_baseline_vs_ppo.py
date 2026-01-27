"""
compare_baseline_vs_ppo.py

Comparison script for baseline policies vs trained PPO.

Generates:
- Reward comparison plots
- Service level comparison
- Stockout comparison
- Cost analysis

Demonstrates learning improvement over rule-based baseline.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.hmarl_env import HMARLEnvironment, SingleAgentWrapper
import config.simulation_config as config

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


class BaselineVsPPOComparator:
    """
    Compare baseline rule-based policy vs trained PPO.
    
    Evaluates both policies on same scenarios and generates
    comparison plots.
    """
    
    def __init__(
        self,
        env_config: Dict,
        ppo_model_path: str,
        n_eval_episodes: int = 20,
        output_dir: str = 'comparison_outputs'
    ):
        """
        Initialize comparator.
        
        Args:
            env_config: Environment configuration
            ppo_model_path: Path to trained PPO model
            n_eval_episodes: Number of evaluation episodes
            output_dir: Output directory for plots
        """
        self.env_config = env_config
        self.ppo_model_path = ppo_model_path
        self.n_eval_episodes = n_eval_episodes
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Results storage
        self.baseline_results = None
        self.ppo_results = None
    
    def evaluate_baseline(self) -> Dict:
        """
        Evaluate baseline rule-based policy.
        
        Returns:
            Evaluation metrics
        """
        print("="*60)
        print("EVALUATING BASELINE POLICY")
        print("="*60)
        
        # Create environment
        multi_env = HMARLEnvironment(config=self.env_config, max_steps=30)
        env = SingleAgentWrapper(multi_env, agent_id='store_store_1')
        
        rewards = []
        service_levels = []
        stockouts = []
        holding_costs = []
        
        for episode in range(self.n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Use agent's default rule-based policy
                action = env.env.agents['store_store_1'].act(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
            
            # Extract metrics
            if 'reconciliation' in info:
                recon = info['reconciliation']
                
                for agent_id, metrics in recon.items():
                    if 'store' in agent_id:
                        service_levels.append(metrics.get('service_level', 0))
                        stockouts.append(metrics.get('lost_sales', 0))
                        holding_costs.append(metrics.get('holding_cost', 0))
            
            if (episode + 1) % 5 == 0:
                print(f"Episode {episode + 1}/{self.n_eval_episodes}: Reward = {episode_reward:.2f}")
        
        env.close()
        
        results = {
            'rewards': rewards,
            'service_levels': service_levels,
            'stockouts': stockouts,
            'holding_costs': holding_costs
        }
        
        print(f"\nBaseline Results:")
        print(f"  Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Avg Service Level: {np.mean(service_levels):.3f}")
        print(f"  Avg Stockouts: {np.mean(stockouts):.1f}")
        print(f"  Avg Holding Cost: ${np.mean(holding_costs):.2f}")
        
        self.baseline_results = results
        return results
    
    def evaluate_ppo(self) -> Dict:
        """
        Evaluate trained PPO policy.
        
        Returns:
            Evaluation metrics
        """
        if not SB3_AVAILABLE:
            raise ImportError("Stable-Baselines3 required")
        
        print("\n" + "="*60)
        print("EVALUATING PPO POLICY")
        print("="*60)
        
        # Load model
        model = PPO.load(self.ppo_model_path)
        print(f"Loaded model from: {self.ppo_model_path}")
        
        # Create environment
        multi_env = HMARLEnvironment(config=self.env_config, max_steps=30)
        env = SingleAgentWrapper(multi_env, agent_id='store_store_1')
        
        rewards = []
        service_levels = []
        stockouts = []
        holding_costs = []
        
        for episode in range(self.n_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
            
            rewards.append(episode_reward)
            
            # Extract metrics
            if 'reconciliation' in info:
                recon = info['reconciliation']
                
                for agent_id, metrics in recon.items():
                    if 'store' in agent_id:
                        service_levels.append(metrics.get('service_level', 0))
                        stockouts.append(metrics.get('lost_sales', 0))
                        holding_costs.append(metrics.get('holding_cost', 0))
            
            if (episode + 1) % 5 == 0:
                print(f"Episode {episode + 1}/{self.n_eval_episodes}: Reward = {episode_reward:.2f}")
        
        env.close()
        
        results = {
            'rewards': rewards,
            'service_levels': service_levels,
            'stockouts': stockouts,
            'holding_costs': holding_costs
        }
        
        print(f"\nPPO Results:")
        print(f"  Avg Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Avg Service Level: {np.mean(service_levels):.3f}")
        print(f"  Avg Stockouts: {np.mean(stockouts):.1f}")
        print(f"  Avg Holding Cost: ${np.mean(holding_costs):.2f}")
        
        self.ppo_results = results
        return results
    
    def generate_comparison_plots(self):
        """Generate comparison plots."""
        if self.baseline_results is None or self.ppo_results is None:
            raise ValueError("Must run evaluations first")
        
        print(f"\n{'='*60}")
        print("GENERATING COMPARISON PLOTS")
        print(f"{'='*60}")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Baseline vs PPO Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Rewards
        ax = axes[0, 0]
        baseline_rewards = self.baseline_results['rewards']
        ppo_rewards = self.ppo_results['rewards']
        
        x = np.arange(len(baseline_rewards))
        width = 0.35
        
        ax.bar(x - width/2, baseline_rewards, width, label='Baseline', alpha=0.7, color='blue')
        ax.bar(x + width/2, ppo_rewards, width, label='PPO', alpha=0.7, color='green')
        
        ax.axhline(y=np.mean(baseline_rewards), color='blue', linestyle='--', alpha=0.5, label='Baseline Avg')
        ax.axhline(y=np.mean(ppo_rewards), color='green', linestyle='--', alpha=0.5, label='PPO Avg')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards: Baseline vs PPO')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Service Level
        ax = axes[0, 1]
        
        baseline_sl = self.baseline_results['service_levels']
        ppo_sl = self.ppo_results['service_levels']
        
        data = [baseline_sl, ppo_sl]
        labels = ['Baseline', 'PPO']
        colors = ['blue', 'green']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axhline(y=0.95, color='red', linestyle='--', label='Target (95%)', linewidth=2)
        ax.set_ylabel('Service Level')
        ax.set_title('Service Level Distribution')
        ax.set_ylim([0.8, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Stockouts
        ax = axes[1, 0]
        
        baseline_stockouts = self.baseline_results['stockouts']
        ppo_stockouts = self.ppo_results['stockouts']
        
        data = [baseline_stockouts, ppo_stockouts]
        labels = ['Baseline', 'PPO']
        colors = ['blue', 'green']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Stockouts')
        ax.set_title('Stockout Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean values as text
        baseline_mean = np.mean(baseline_stockouts)
        ppo_mean = np.mean(ppo_stockouts)
        ax.text(1, baseline_mean, f'{baseline_mean:.1f}', ha='center', va='bottom', fontweight='bold')
        ax.text(2, ppo_mean, f'{ppo_mean:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Holding Costs
        ax = axes[1, 1]
        
        baseline_costs = self.baseline_results['holding_costs']
        ppo_costs = self.ppo_results['holding_costs']
        
        data = [baseline_costs, ppo_costs]
        labels = ['Baseline', 'PPO']
        colors = ['blue', 'green']
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Holding Cost ($)')
        ax.set_title('Holding Cost Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean values as text
        baseline_mean = np.mean(baseline_costs)
        ppo_mean = np.mean(ppo_costs)
        ax.text(1, baseline_mean, f'${baseline_mean:.0f}', ha='center', va='bottom', fontweight='bold')
        ax.text(2, ppo_mean, f'${ppo_mean:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'baseline_vs_ppo_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plots saved to: {plot_path}")
        
        plt.close()
    
    def print_summary(self):
        """Print comparison summary."""
        if self.baseline_results is None or self.ppo_results is None:
            return
        
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        # Rewards
        baseline_reward = np.mean(self.baseline_results['rewards'])
        ppo_reward = np.mean(self.ppo_results['rewards'])
        reward_improvement = ((ppo_reward - baseline_reward) / abs(baseline_reward)) * 100
        
        print(f"\nRewards:")
        print(f"  Baseline: {baseline_reward:.2f}")
        print(f"  PPO:      {ppo_reward:.2f}")
        print(f"  Improvement: {reward_improvement:+.1f}%")
        
        # Service Level
        baseline_sl = np.mean(self.baseline_results['service_levels'])
        ppo_sl = np.mean(self.ppo_results['service_levels'])
        sl_improvement = ((ppo_sl - baseline_sl) / baseline_sl) * 100
        
        print(f"\nService Level:")
        print(f"  Baseline: {baseline_sl:.3f}")
        print(f"  PPO:      {ppo_sl:.3f}")
        print(f"  Improvement: {sl_improvement:+.1f}%")
        
        # Stockouts
        baseline_stockouts = np.mean(self.baseline_results['stockouts'])
        ppo_stockouts = np.mean(self.ppo_results['stockouts'])
        stockout_reduction = ((baseline_stockouts - ppo_stockouts) / baseline_stockouts) * 100 if baseline_stockouts > 0 else 0
        
        print(f"\nStockouts:")
        print(f"  Baseline: {baseline_stockouts:.1f}")
        print(f"  PPO:      {ppo_stockouts:.1f}")
        print(f"  Reduction: {stockout_reduction:.1f}%")
        
        # Holding Costs
        baseline_costs = np.mean(self.baseline_results['holding_costs'])
        ppo_costs = np.mean(self.ppo_results['holding_costs'])
        cost_reduction = ((baseline_costs - ppo_costs) / baseline_costs) * 100
        
        print(f"\nHolding Costs:")
        print(f"  Baseline: ${baseline_costs:.2f}")
        print(f"  PPO:      ${ppo_costs:.2f}")
        print(f"  Reduction: {cost_reduction:.1f}%")
        
        # Save summary to JSON
        summary = {
            'rewards': {
                'baseline': float(baseline_reward),
                'ppo': float(ppo_reward),
                'improvement_pct': float(reward_improvement)
            },
            'service_level': {
                'baseline': float(baseline_sl),
                'ppo': float(ppo_sl),
                'improvement_pct': float(sl_improvement)
            },
            'stockouts': {
                'baseline': float(baseline_stockouts),
                'ppo': float(ppo_stockouts),
                'reduction_pct': float(stockout_reduction)
            },
            'holding_costs': {
                'baseline': float(baseline_costs),
                'ppo': float(ppo_costs),
                'reduction_pct': float(cost_reduction)
            }
        }
        
        summary_path = os.path.join(self.output_dir, 'comparison_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")
    
    def run_comparison(self):
        """Run full comparison pipeline."""
        # Evaluate baseline
        self.evaluate_baseline()
        
        # Evaluate PPO
        self.evaluate_ppo()
        
        # Generate plots
        self.generate_comparison_plots()
        
        # Print summary
        self.print_summary()


def main():
    """Main comparison runner."""
    # Environment configuration
    env_config = {
        'SIMULATION_DAYS': 30,
        'WARMUP_DAYS': 0,
        'RANDOM_SEED': 123,  # Different seed for evaluation
        'SKU_CONFIG': config.SKU_CONFIG,
        'STORE_CONFIG': config.STORE_CONFIG,
        'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
        'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
    }
    
    # PPO model path
    ppo_model_path = 'training_outputs/phase1/ppo_store_agents_phase1.zip'
    
    if not os.path.exists(ppo_model_path):
        print(f"ERROR: PPO model not found at {ppo_model_path}")
        print(f"Please run train_ppo_phase1.py first")
        return 1
    
    # Create comparator
    comparator = BaselineVsPPOComparator(
        env_config=env_config,
        ppo_model_path=ppo_model_path,
        n_eval_episodes=20,
        output_dir='comparison_outputs'
    )
    
    # Run comparison
    comparator.run_comparison()
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code if exit_code else 0)
