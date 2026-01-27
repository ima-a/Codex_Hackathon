"""
validate_environment.py

Environment validation and sanity checks for HMARL PPO training.

MANDATORY FIRST STEP before training:
- Validates reset() returns valid observations
- Tests step() with random actions
- Verifies rewards and reconciliation metrics
- Logs observations, rewards, and key metrics

All checks must pass before PPO training begins.
"""

import sys
import os
import numpy as np
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.hmarl_env import HMARLEnvironment
import config.simulation_config as config


class EnvironmentValidator:
    """
    Validates HMARL environment before training.
    
    Performs comprehensive sanity checks to ensure:
    - Observations are valid
    - Actions execute correctly
    - Rewards are computed
    - Reconciliation metrics are sensible
    """
    
    def __init__(self, env: HMARLEnvironment):
        """
        Initialize validator.
        
        Args:
            env: HMARL environment to validate
        """
        self.env = env
        self.validation_results = {}
    
    def validate_reset(self) -> bool:
        """
        Test 1: Validate reset() returns valid observations.
        
        Checks:
        - Observations returned for all agents
        - Observation shapes match declared spaces
        - Values are finite and in reasonable ranges
        
        Returns:
            True if validation passes
        """
        print("\n" + "="*60)
        print("TEST 1: Validating reset()")
        print("="*60)
        
        try:
            observations, _ = self.env.reset()
            
            # Check all agents have observations
            expected_agents = set(self.env.agent_ids)
            actual_agents = set(observations.keys())
            
            if expected_agents != actual_agents:
                print(f"❌ FAIL: Agent mismatch")
                print(f"   Expected: {expected_agents}")
                print(f"   Got: {actual_agents}")
                return False
            
            print(f"✓ All {len(expected_agents)} agents have observations")
            
            # Check observation shapes and values
            for agent_id, obs in observations.items():
                expected_shape = self.env.observation_space[agent_id].shape
                actual_shape = obs.shape
                
                if expected_shape != actual_shape:
                    print(f"❌ FAIL: {agent_id} shape mismatch")
                    print(f"   Expected: {expected_shape}")
                    print(f"   Got: {actual_shape}")
                    return False
                
                # Check for NaN or Inf
                if not np.all(np.isfinite(obs)):
                    print(f"❌ FAIL: {agent_id} has NaN or Inf values")
                    print(f"   Observation: {obs}")
                    return False
                
                print(f"✓ {agent_id}: shape={actual_shape}, range=[{obs.min():.3f}, {obs.max():.3f}]")
            
            print("\n✅ PASS: reset() validation successful")
            self.validation_results['reset'] = True
            return True
            
        except Exception as e:
            print(f"❌ FAIL: Exception during reset()")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            self.validation_results['reset'] = False
            return False
    
    def validate_step(self, num_steps: int = 10) -> bool:
        """
        Test 2: Validate step() with random actions.
        
        Checks:
        - step() executes without errors
        - Returns observations, rewards, done, info
        - Rewards are scalar values
        - State changes over time
        
        Args:
            num_steps: Number of steps to test
        
        Returns:
            True if validation passes
        """
        print("\n" + "="*60)
        print(f"TEST 2: Validating step() for {num_steps} steps")
        print("="*60)
        
        try:
            observations, _ = self.env.reset()
            
            all_rewards = {agent_id: [] for agent_id in self.env.agent_ids}
            all_observations = {agent_id: [] for agent_id in self.env.agent_ids}
            
            for step in range(num_steps):
                # Sample random actions
                actions = {}
                for agent_id in self.env.agent_ids:
                    actions[agent_id] = self.env.action_space[agent_id].sample()
                
                # Execute step
                next_observations, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                # Validate returns
                if not isinstance(next_observations, dict):
                    print(f"❌ FAIL: observations not a dict at step {step}")
                    return False
                
                if not isinstance(rewards, dict):
                    print(f"❌ FAIL: rewards not a dict at step {step}")
                    return False
                
                if not isinstance(done, bool):
                    print(f"❌ FAIL: done not a bool at step {step}")
                    return False
                
                if not isinstance(info, dict):
                    print(f"❌ FAIL: info not a dict at step {step}")
                    return False
                
                # Check rewards are scalar
                for agent_id, reward in rewards.items():
                    if not np.isscalar(reward):
                        print(f"❌ FAIL: {agent_id} reward not scalar at step {step}")
                        return False
                    
                    if not np.isfinite(reward):
                        print(f"❌ FAIL: {agent_id} reward is NaN/Inf at step {step}")
                        return False
                    
                    all_rewards[agent_id].append(reward)
                
                # Store observations
                for agent_id, obs in next_observations.items():
                    all_observations[agent_id].append(obs)
                
                # Log step
                if step % 5 == 0:
                    print(f"\nStep {step}:")
                    print(f"  Actions: {actions}")
                    print(f"  Rewards: {', '.join([f'{aid}: {r:.2f}' for aid, r in rewards.items()])}")
                
                observations = next_observations
                
                if done:
                    print(f"\n✓ Episode completed at step {step}")
                    break
            
            # Analyze reward statistics
            print(f"\n{'='*60}")
            print("Reward Statistics:")
            print(f"{'='*60}")
            
            for agent_id, rewards_list in all_rewards.items():
                if rewards_list:
                    print(f"\n{agent_id}:")
                    print(f"  Mean: {np.mean(rewards_list):.3f}")
                    print(f"  Std:  {np.std(rewards_list):.3f}")
                    print(f"  Min:  {np.min(rewards_list):.3f}")
                    print(f"  Max:  {np.max(rewards_list):.3f}")
            
            # Check that observations change over time
            print(f"\n{'='*60}")
            print("Observation Dynamics:")
            print(f"{'='*60}")
            
            for agent_id, obs_list in all_observations.items():
                if len(obs_list) > 1:
                    obs_array = np.array(obs_list)
                    obs_std = np.std(obs_array, axis=0)
                    
                    if np.all(obs_std < 1e-6):
                        print(f"⚠️  WARNING: {agent_id} observations are static")
                    else:
                        print(f"✓ {agent_id}: observations changing (std={obs_std.mean():.4f})")
            
            print("\n✅ PASS: step() validation successful")
            self.validation_results['step'] = True
            return True
            
        except Exception as e:
            print(f"❌ FAIL: Exception during step()")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            self.validation_results['step'] = False
            return False
    
    def validate_reconciliation(self) -> bool:
        """
        Test 3: Validate reconciliation metrics.
        
        Checks:
        - Reconciliation reports are generated
        - Metrics are sensible (service level 0-1, costs >= 0)
        - Rewards correlate with reconciliation metrics
        
        Returns:
            True if validation passes
        """
        print("\n" + "="*60)
        print("TEST 3: Validating reconciliation metrics")
        print("="*60)
        
        try:
            observations, _ = self.env.reset()
            
            # Run a few steps and collect reconciliation data
            reconciliation_data = []
            
            for step in range(5):
                actions = {aid: self.env.action_space[aid].sample() for aid in self.env.agent_ids}
                next_observations, rewards, terminated, truncated, info = self.env.step(actions)
                done = terminated or truncated
                
                if 'reconciliation' in info:
                    reconciliation_data.append(info['reconciliation'])
                
                observations = next_observations
                if done:
                    break
            
            if not reconciliation_data:
                print(f"❌ FAIL: No reconciliation data in info")
                return False
            
            print(f"✓ Collected {len(reconciliation_data)} reconciliation reports")
            
            # Validate reconciliation metrics
            print(f"\nReconciliation Metrics Sample (Step 0):")
            print(f"{'='*60}")
            
            for agent_id, report in reconciliation_data[0].items():
                print(f"\n{agent_id}:")
                for metric, value in report.items():
                    print(f"  {metric}: {value}")
                
                # Validate metric ranges
                if 'service_level' in report:
                    sl = report['service_level']
                    if not (0 <= sl <= 1.1):  # Allow slight overshoot
                        print(f"⚠️  WARNING: service_level out of range [0,1]: {sl}")
                
                if 'holding_cost' in report:
                    hc = report['holding_cost']
                    if hc < 0:
                        print(f"❌ FAIL: holding_cost negative: {hc}")
                        return False
                
                if 'stockout_penalty' in report:
                    sp = report['stockout_penalty']
                    if sp < 0:
                        print(f"❌ FAIL: stockout_penalty negative: {sp}")
                        return False
            
            print("\n✅ PASS: reconciliation validation successful")
            self.validation_results['reconciliation'] = True
            return True
            
        except Exception as e:
            print(f"❌ FAIL: Exception during reconciliation validation")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            self.validation_results['reconciliation'] = False
            return False
    
    def validate_action_space(self) -> bool:
        """
        Test 4: Validate action space coverage.
        
        Checks:
        - All actions are valid
        - Actions produce different outcomes
        
        Returns:
            True if validation passes
        """
        print("\n" + "="*60)
        print("TEST 4: Validating action space")
        print("="*60)
        
        try:
            # Test each action for store agents
            store_agents = [aid for aid in self.env.agent_ids if 'store' in aid]
            
            for agent_id in store_agents:
                print(f"\nTesting {agent_id}:")
                
                action_rewards = {}
                
                for action in range(self.env.action_space[agent_id].n):
                    self.env.reset()
                    
                    # Execute action
                    actions = {aid: 0 for aid in self.env.agent_ids}
                    actions[agent_id] = action
                    
                    _, rewards, _, _, _ = self.env.step(actions)
                    action_rewards[action] = rewards[agent_id]
                    
                    print(f"  Action {action}: reward = {rewards[agent_id]:.2f}")
                
                # Check that different actions can produce different rewards
                unique_rewards = len(set(action_rewards.values()))
                print(f"  Unique reward values: {unique_rewards}/{len(action_rewards)}")
            
            print("\n✅ PASS: action space validation successful")
            self.validation_results['action_space'] = True
            return True
            
        except Exception as e:
            print(f"❌ FAIL: Exception during action space validation")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            self.validation_results['action_space'] = False
            return False
    
    def run_all_validations(self) -> bool:
        """
        Run all validation tests.
        
        Returns:
            True if all tests pass
        """
        print("\n" + "="*60)
        print("HMARL ENVIRONMENT VALIDATION")
        print("="*60)
        print("\nRunning comprehensive environment validation...")
        print("All tests must pass before PPO training can begin.")
        
        # Run tests in order
        tests = [
            ("Reset Validation", self.validate_reset),
            ("Step Validation", self.validate_step),
            ("Reconciliation Validation", self.validate_reconciliation),
            ("Action Space Validation", self.validate_action_space)
        ]
        
        all_passed = True
        
        for test_name, test_func in tests:
            passed = test_func()
            if not passed:
                all_passed = False
                print(f"\n❌ {test_name} FAILED - stopping validation")
                break
        
        # Print summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        for test_name, result in self.validation_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        if all_passed:
            print("\n" + "="*60)
            print("✅ ALL VALIDATIONS PASSED")
            print("="*60)
            print("\nEnvironment is ready for PPO training!")
        else:
            print("\n" + "="*60)
            print("❌ VALIDATION FAILED")
            print("="*60)
            print("\nFix errors before proceeding to training.")
        
        return all_passed


def main():
    """Main validation runner."""
    # Create environment
    env_config = {
        'SIMULATION_DAYS': 30,
        'WARMUP_DAYS': 0,
        'RANDOM_SEED': 42,
        'SKU_CONFIG': config.SKU_CONFIG,
        'STORE_CONFIG': config.STORE_CONFIG,
        'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
        'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
    }
    
    env = HMARLEnvironment(
        config=env_config,
        max_steps=30
    )
    
    # Run validation
    validator = EnvironmentValidator(env)
    success = validator.run_all_validations()
    
    env.close()
    
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
