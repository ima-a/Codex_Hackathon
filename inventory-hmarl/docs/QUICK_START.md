# 🚀 HMARL Quick Start Guide

**Get started with the Hierarchical Multi-Agent RL system in 5 minutes**

---

## 📋 Prerequisites

```bash
# Required packages
pip install numpy torch gym
```

---

## ⚡ Quick Training (3 Steps)

### Step 1: Navigate to project
```bash
cd /home/Ima/work/hackathon/codex/inventory-hmarl
```

### Step 2: Run training script
```bash
python agents/train_with_gym_env.py
```

### Step 3: Check results
```bash
ls checkpoints/
# Output: ppo_store_agents_gym.pt
```

**Done!** You've trained store agents with PPO.

---

## 🎯 System Overview

```
Digital Twin → Agents → Actions → Reconciliation → Rewards → Learning
     ↑                                                           ↓
     └───────────────────── Updated Policy ─────────────────────┘
```

**Agents:**
- **Store Agents (×2):** Learning-enabled (PPO)
- **Warehouse Agent:** Rule-based
- **Supplier Agent:** Rule-based

**Learning:**
- Algorithm: PPO (Proximal Policy Optimization)
- Architecture: CTDE (Centralized Training, Decentralized Execution)
- Parameter Sharing: Store agents share policy

---

## 📝 Basic Usage

### Create Environment

```python
from env.hmarl_env import HMARLEnvironment
import config.simulation_config as config

env = HMARLEnvironment(
    config={
        'SIMULATION_DAYS': 30,
        'RANDOM_SEED': 42,
        'SKU_CONFIG': config.SKU_CONFIG,
        'STORE_CONFIG': config.STORE_CONFIG,
        'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
        'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
    },
    max_steps=30
)
```

### Run Episode

```python
# Reset
observations = env.reset()

# Step
actions = {
    'store_store_1': 2,  # order_1.0x
    'store_store_2': 1,  # order_0.5x
    'warehouse_warehouse_1': 2,  # order_medium
    'supplier_supplier_1': 0   # fulfill_full
}

next_obs, rewards, done, info = env.step(actions)

print(f"Rewards: {rewards}")
# Output: {'store_store_1': 8.5, 'store_store_2': 7.2, ...}
```

---

## 🧠 Agent Actions

### Store Agents (4 actions)
```python
0: no_order       # Don't order
1: order_0.5x     # Order 0.5 × forecast
2: order_1.0x     # Order 1.0 × forecast
3: order_1.5x     # Order 1.5 × forecast
```

### Warehouse Agent (4 actions)
```python
0: no_order       # Don't order
1: order_low      # Order 500 units
2: order_medium   # Order 1000 units
3: order_high     # Order 1500 units
```

### Supplier Agent (3 actions)
```python
0: fulfill_full    # Ship 100%
1: fulfill_partial # Ship 80%
2: delay           # Delay shipment
```

---

## 📊 Observation Spaces

### Store Agent (7D)
```python
[
    current_inventory,         # 0-1 (normalized)
    forecasted_demand,         # 0-1
    demand_uncertainty,        # 0-1
    days_of_inventory_cover,   # 0-1
    last_day_stockout,         # 0 or 1
    warehouse_inventory_ratio, # 0-1
    recent_service_level       # 0-1
]
```

### Warehouse Agent (6D)
```python
[
    aggregate_store_demand,
    current_warehouse_inventory,
    inbound_supplier_pipeline,
    store_1_inventory,
    store_2_inventory,
    avg_store_service_level
]
```

### Supplier Agent (3D)
```python
[
    warehouse_order_quantity,
    current_production_capacity,
    days_until_delivery
]
```

---

## 💰 Reward Functions

### Store Agent
```python
reward = (
    +10.0 * service_level      # Maximize service
    - 0.1 * holding_cost       # Minimize inventory
    - 5.0 * stockout_penalty   # Avoid stockouts
    - 0.05 * excess_inventory  # Avoid over-ordering
)
```

### Warehouse Agent
```python
reward = (
    +5.0 * avg_store_service_level
    - 0.05 * warehouse_holding_cost
    - 3.0 * store_stockout_count
)
```

### Supplier Agent
```python
reward = (
    +2.0 * fulfillment_rate
    - 1.0 * delay_penalty
)
```

---

## 🎓 Training Examples

### Example 1: Train Store Agents Only

```python
from env.hmarl_env import HMARLEnvironment
from agents.ppo_trainer import PPOTrainer
import config.simulation_config as config

# Setup
env = HMARLEnvironment(config={...}, max_steps=30)
ppo = PPOTrainer(obs_dim=7, action_dim=4)

# Train
for episode in range(100):
    obs = env.reset()
    experiences = []
    
    for step in range(30):
        actions = {}
        for agent_id in env.agent_ids:
            if 'store' in agent_id:
                actions[agent_id] = ppo.select_action(obs[agent_id])
            else:
                actions[agent_id] = env.agents[agent_id].act(obs[agent_id])
        
        next_obs, rewards, done, info = env.step(actions)
        
        for agent_id in ['store_store_1', 'store_store_2']:
            experiences.append({
                'observation': obs[agent_id],
                'action': actions[agent_id],
                'reward': rewards[agent_id],
                'done': done
            })
        
        obs = next_obs
        if done: break
    
    ppo.update(experiences)

ppo.save_checkpoint('model.pt')
```

### Example 2: Evaluate Trained Policy

```python
# Load trained model
ppo.load_checkpoint('model.pt')

# Evaluate
env = HMARLEnvironment(config={...}, max_steps=30)
obs = env.reset()

total_rewards = {aid: 0 for aid in env.agent_ids}

for step in range(30):
    actions = {}
    for agent_id in env.agent_ids:
        if 'store' in agent_id:
            actions[agent_id] = ppo.select_action(obs[agent_id], deterministic=True)
        else:
            actions[agent_id] = env.agents[agent_id].act(obs[agent_id])
    
    obs, rewards, done, info = env.step(actions)
    
    for agent_id in env.agent_ids:
        total_rewards[agent_id] += rewards[agent_id]
    
    if done: break

print(f"Total Rewards: {total_rewards}")
```

---

## 🔧 Configuration

### Modify Agent Behavior

```python
agent_configs = {
    'store_store_1': {
        'reorder_point': 300,      # Trigger level
        'order_up_to_level': 700,  # Target level
        'forecast_window': 7       # Days for forecast
    },
    'warehouse_warehouse_1': {
        'reorder_point': 2000,
        'order_up_to_level': 4000
    }
}

env = HMARLEnvironment(
    config=env_config,
    agent_configs=agent_configs,
    max_steps=30
)
```

### Modify PPO Hyperparameters

```python
ppo = PPOTrainer(
    obs_dim=7,
    action_dim=4,
    hidden_dim=64,        # Network size
    lr=3e-4,              # Learning rate
    gamma=0.99,           # Discount factor
    gae_lambda=0.95,      # GAE parameter
    clip_epsilon=0.2,     # PPO clip
    entropy_coef=0.01     # Exploration bonus
)
```

---

## 📈 Monitoring Training

### Print Episode Stats

```python
for episode in range(100):
    # ... training code ...
    
    stats = env.get_episode_stats()
    print(f"Episode {episode}:")
    for agent_id, agent_stats in stats.items():
        print(f"  {agent_id}: {agent_stats['total_reward']:.2f}")
```

### Track PPO Metrics

```python
trainer_stats = ppo.get_stats()
print(f"Policy Loss: {trainer_stats['avg_policy_loss']:.4f}")
print(f"Value Loss: {trainer_stats['avg_value_loss']:.4f}")
```

---

## 🐛 Troubleshooting

### Issue: Import errors
```bash
# Make sure you're in the project root
cd /home/Ima/work/hackathon/codex/inventory-hmarl
python agents/train_with_gym_env.py
```

### Issue: CUDA errors
```python
# Use CPU instead
ppo = PPOTrainer(..., device='cpu')
```

### Issue: Low rewards
- Increase training episodes (100+)
- Adjust reward coefficients
- Tune PPO hyperparameters
- Check reconciliation metrics

---

## 📚 File Reference

```
agents/
├── base_agent.py          # Agent interface
├── store_agent.py         # Store agent implementation
├── warehouse_agent.py     # Warehouse agent
├── supplier_agent.py      # Supplier agent
├── learning_wrapper.py    # Learning wrapper
├── ppo_trainer.py         # PPO algorithm
├── train_hmarl.py         # Manual training
└── train_with_gym_env.py  # Gym-based training

env/
├── digital_twin.py        # Simulation environment
└── hmarl_env.py          # Gym wrapper

docs/
└── HMARL_ARCHITECTURE.md  # Full documentation
```

---

## ✅ Checklist

Before training:
- [ ] Environment configured
- [ ] Agents initialized
- [ ] PPO trainer created
- [ ] Checkpoint directory exists

After training:
- [ ] Model saved to checkpoints/
- [ ] Training stats logged
- [ ] Evaluation completed
- [ ] Performance compared to baseline

---

## 🎯 Expected Results

**After 20 episodes:**
- Store agents learn basic ordering patterns
- Rewards increase from ~5 to ~8 per step
- Service levels remain high (>95%)

**After 100 episodes:**
- Near-optimal ordering policies
- Rewards stabilize around 8-10 per step
- Better cost-service trade-off than baseline

---

## 🚀 Next Steps

1. **Run baseline comparison:**
   ```bash
   python baseline_policies/baseline_runner.py
   ```

2. **Train for more episodes:**
   - Modify `num_episodes=100` in training script

3. **Experiment with scenarios:**
   - Demand spike
   - Supply disruption
   - Seasonal variation

4. **Visualize results:**
   - Add tensorboard logging
   - Plot reward curves
   - Compare policies

---

**Ready to train?** Run: `python agents/train_with_gym_env.py`

---

*Quick Start Guide - HMARL System*  
*Last Updated: January 28, 2026*
