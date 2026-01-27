# 🎓 PPO Training Pipeline Documentation

**Phase-1 Training for HMARL System**

---

## 📋 Overview

This document explains the complete PPO training pipeline for the Hierarchical Multi-Agent Reinforcement Learning (HMARL) system.

### Training Scope: Phase-1

**What is trained:**
- ✅ 2 Store agents (shared PPO policy)

**What remains rule-based:**
- ⏸️ Warehouse agent (deterministic)
- ⏸️ Supplier agent (deterministic)

**Architecture:**
- Centralized Training, Decentralized Execution (CTDE)
- Parameter sharing across store agents
- Reconciliation-driven rewards

---

## 🔄 Complete Training Workflow

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Environment Validation (MANDATORY)             │
│  validate_environment.py                                │
│  - Test reset() and step()                              │
│  - Verify observations and rewards                      │
│  - Check reconciliation metrics                         │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼ (All tests pass)
┌─────────────────────────────────────────────────────────┐
│  STEP 2: PPO Training                                   │
│  train_ppo_phase1.py                                    │
│  - Train store agents with shared policy                │
│  - Collect metrics during training                      │
│  - Generate training plots                              │
│  - Save trained model                                   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: Baseline Comparison                            │
│  compare_baseline_vs_ppo.py                             │
│  - Evaluate baseline rule-based policy                  │
│  - Evaluate trained PPO policy                          │
│  - Generate comparison plots                            │
│  - Statistical analysis                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install stable-baselines3 matplotlib numpy gym

# Navigate to project
cd /home/Ima/work/hackathon/codex/inventory-hmarl
```

### Step 1: Validate Environment (MANDATORY)

```bash
python training/validate_environment.py
```

**Expected output:**
```
✅ PASS: reset() validation successful
✅ PASS: step() validation successful
✅ PASS: reconciliation validation successful
✅ PASS: action space validation successful

✅ ALL VALIDATIONS PASSED
Environment is ready for PPO training!
```

**Do NOT proceed to training if validation fails!**

### Step 2: Train PPO

```bash
python training/train_ppo_phase1.py
```

**Expected output:**
```
PHASE-1 PPO TRAINING
Training Configuration:
  Total timesteps: 10,000
  Learning rate: 0.0003
  ...

Starting training...
[Progress bar]

Training complete!
Model saved to: training_outputs/phase1/ppo_store_agents_phase1.zip
Training plots saved to: training_outputs/phase1/training_plots.png
```

### Step 3: Compare vs Baseline

```bash
python training/compare_baseline_vs_ppo.py
```

**Expected output:**
```
COMPARISON SUMMARY
Rewards:
  Baseline: 150.23
  PPO:      178.45
  Improvement: +18.8%

Service Level:
  Baseline: 0.952
  PPO:      0.968
  Improvement: +1.7%
...
```

---

## 📊 Data Flow Architecture

### 1. Environment → Observation

```python
# Multi-agent environment state
state = {
    'stores': {
        'store_1': {inventory: {...}, demand: {...}, ...},
        'store_2': {inventory: {...}, demand: {...}, ...}
    },
    'warehouses': {...},
    'suppliers': {...}
}

# Each agent extracts local observation
store_1_obs = store_agent_1.observe(state)
# → [inventory, forecast, uncertainty, days_cover, stockout, wh_ratio, service_level]
```

### 2. Observation → Action (PPO)

```python
# PPO policy network
action_logits = policy_network(observation)
action_distribution = Categorical(logits=action_logits)
action = action_distribution.sample()

# Actions:
# 0: no_order
# 1: order_0.5x
# 2: order_1.0x
# 3: order_1.5x
```

### 3. Action → Environment Step

```python
# All agents act
actions = {
    'store_store_1': ppo_action,      # PPO-selected
    'store_store_2': ppo_action,      # PPO-selected (shared policy)
    'warehouse_warehouse_1': rule_action,  # Rule-based
    'supplier_supplier_1': rule_action     # Rule-based
}

# Environment executes
next_state, rewards, done, info = env.step(actions)
```

### 4. Reconciliation → Reward

```python
# Reconciliation engine computes metrics
reconciliation_report = {
    'service_level': 0.95,
    'holding_cost': 45.2,
    'stockout_penalty': 0.0,
    'excess_inventory': 120
}

# Agent computes reward
reward = (
    +10.0 * service_level       # +9.5
    - 0.1 * holding_cost        # -4.52
    - 5.0 * stockout_penalty    # -0.0
    - 0.05 * excess_inventory   # -6.0
)
# Total reward = -1.02
```

### 5. Experience → Learning

```python
# Experience tuple
experience = {
    'observation': obs,
    'action': action,
    'reward': reward,
    'next_observation': next_obs,
    'done': done
}

# PPO update (after collecting batch)
# 1. Compute advantages using GAE
# 2. Update policy with clipped objective
# 3. Update value function
# 4. Apply entropy bonus
```

---

## 🎯 Why This Qualifies as HMARL

### Hierarchical

**Hierarchy Levels:**
1. **Store Level** (Operational)
   - Local inventory decisions
   - Immediate customer service
   - Short-term optimization

2. **Warehouse Level** (Tactical)
   - Aggregate demand management
   - Multi-store coordination
   - Medium-term planning

3. **Supplier Level** (Strategic)
   - Production scheduling
   - Lead time management
   - Long-term capacity

**Hierarchical Coordination:**
- Store agents optimize local service vs cost
- Warehouse agent balances store service levels
- Supplier agent ensures upstream reliability

### Multi-Agent

**Multiple Autonomous Agents:**
- 2 Store agents (learning)
- 1 Warehouse agent (rule-based, upgradeable)
- 1 Supplier agent (rule-based)

**Decentralized Execution:**
- Each agent observes local state
- Each agent acts independently
- No direct communication during execution

**Centralized Training:**
- Store agents share policy network
- Experiences pooled for training
- Coordinated learning

### Reinforcement Learning

**RL Components:**
- **State:** Multi-echelon supply chain state
- **Actions:** Discrete ordering decisions
- **Rewards:** Reconciliation-derived metrics
- **Policy:** PPO neural network
- **Learning:** Gradient-based optimization

**RL Characteristics:**
- Sequential decision making
- Delayed rewards
- Exploration vs exploitation
- Credit assignment

---

## 🔧 PPO Specification

### Algorithm: Proximal Policy Optimization (PPO)

**Why PPO?**
- ✅ Stable training (clipped objective)
- ✅ Sample efficient
- ✅ Works well with discrete actions
- ✅ Industry-proven (OpenAI, DeepMind)

### Hyperparameters

```python
{
    'learning_rate': 3e-4,      # Adam optimizer learning rate
    'n_steps': 2048,            # Steps per rollout
    'batch_size': 64,           # Minibatch size
    'n_epochs': 10,             # Optimization epochs per rollout
    'gamma': 0.99,              # Discount factor
    'gae_lambda': 0.95,         # GAE lambda
    'clip_range': 0.2,          # PPO clip epsilon
    'ent_coef': 0.01,           # Entropy coefficient
    'vf_coef': 0.5,             # Value function coefficient
    'max_grad_norm': 0.5        # Gradient clipping
}
```

### Network Architecture

**Policy Network (Actor):**
```
Input (7D observation)
    ↓
Linear(7 → 64) + Tanh
    ↓
Linear(64 → 64) + Tanh
    ↓
Linear(64 → 4) [action logits]
    ↓
Softmax → Action probabilities
```

**Value Network (Critic):**
```
Input (7D observation)
    ↓
Linear(7 → 64) + Tanh
    ↓
Linear(64 → 64) + Tanh
    ↓
Linear(64 → 1) [state value]
```

### Parameter Sharing

**Implementation:**
```python
# Single PPO model
ppo_model = PPO("MlpPolicy", env, ...)

# Both store agents use same model
store_1_action = ppo_model.predict(store_1_obs)
store_2_action = ppo_model.predict(store_2_obs)

# Experiences from both agents pooled
experiences = [
    *store_1_experiences,
    *store_2_experiences
]

# Single update for shared policy
ppo_model.learn(experiences)
```

**Benefits:**
- Faster learning (2x data)
- Better generalization
- Consistent behavior across stores

---

## 📈 Expected Results

### Training Progression (10,000 timesteps)

**Episodes 1-50 (Exploration):**
- Rewards: -50 to +50 (high variance)
- Service level: 85-95%
- Stockouts: Frequent
- Policy: Random exploration

**Episodes 50-150 (Learning):**
- Rewards: +50 to +150 (decreasing variance)
- Service level: 90-98%
- Stockouts: Decreasing
- Policy: Learning patterns

**Episodes 150+ (Convergence):**
- Rewards: +150 to +200 (stable)
- Service level: >95%
- Stockouts: Rare
- Policy: Near-optimal

### Baseline Comparison

**Expected Improvements:**
- Rewards: +10% to +30%
- Service level: +1% to +5%
- Stockout reduction: 20% to 50%
- Holding cost reduction: 5% to 15%

**Why PPO Outperforms:**
- Learns demand patterns
- Optimizes cost-service tradeoff
- Adapts to warehouse availability
- Balances exploration-exploitation

---

## 🔮 Extensibility: Future Phases

### Phase-2: Warehouse Agent Training

**Design:**
```python
# Freeze store agents
store_1_agent.freeze()
store_2_agent.freeze()

# Train warehouse agent
warehouse_ppo = PPO(...)
warehouse_ppo.learn(...)

# Unfreeze for joint training (optional)
store_1_agent.unfreeze()
```

**Implementation Points:**
- Warehouse observation space: 6D
- Warehouse action space: 4 discrete
- Reward from reconciliation
- Can reuse same PPO trainer class

### Phase-3: Supplier Agent Training (Optional)

**Design:**
```python
# Freeze downstream agents
freeze_agents(['store_1', 'store_2', 'warehouse_1'])

# Train supplier agent
supplier_ppo = PPO(...)
supplier_ppo.learn(...)
```

### Phase-4: Joint Fine-Tuning

**Design:**
```python
# Unfreeze all agents
unfreeze_all_agents()

# Joint training with reduced learning rate
joint_ppo = PPO(..., learning_rate=1e-4)
joint_ppo.learn(...)
```

### Freeze/Unfreeze Mechanism

```python
class FreezableAgent:
    def __init__(self, agent, learner):
        self.agent = agent
        self.learner = learner
        self.frozen = False
    
    def freeze(self):
        """Freeze policy (no updates)."""
        self.frozen = True
        if self.learner:
            for param in self.learner.policy.parameters():
                param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze policy (allow updates)."""
        self.frozen = False
        if self.learner:
            for param in self.learner.policy.parameters():
                param.requires_grad = True
```

---

## 📁 File Structure

```
training/
├── validate_environment.py       # Step 1: Validation
├── train_ppo_phase1.py           # Step 2: Training
├── compare_baseline_vs_ppo.py    # Step 3: Comparison
└── TRAINING_GUIDE.md             # This file

training_outputs/
└── phase1/
    ├── ppo_store_agents_phase1.zip    # Trained model
    ├── training_metrics.json          # Training data
    ├── training_plots.png             # Visualization
    └── tensorboard/                   # TensorBoard logs

comparison_outputs/
├── baseline_vs_ppo_comparison.png  # Comparison plots
└── comparison_summary.json         # Statistical summary
```

---

## 🐛 Troubleshooting

### Issue: Validation fails

**Solution:**
```bash
# Check environment configuration
python -c "from env.hmarl_env import HMARLEnvironment; import config.simulation_config as config; env = HMARLEnvironment(config={'SKU_CONFIG': config.SKU_CONFIG, 'STORE_CONFIG': config.STORE_CONFIG, 'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG, 'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG, 'SIMULATION_DAYS': 30, 'WARMUP_DAYS': 0, 'RANDOM_SEED': 42}, max_steps=30); print('OK')"
```

### Issue: Training is slow

**Solutions:**
- Reduce `total_timesteps` (e.g., 5000)
- Reduce `n_steps` (e.g., 1024)
- Reduce `n_epochs` (e.g., 5)
- Use GPU if available

### Issue: Rewards not improving

**Solutions:**
- Check reward function coefficients
- Increase `total_timesteps`
- Adjust `learning_rate`
- Verify reconciliation metrics are sensible

### Issue: Model not found for comparison

**Solution:**
```bash
# Ensure training completed successfully
ls training_outputs/phase1/ppo_store_agents_phase1.zip

# If missing, run training again
python training/train_ppo_phase1.py
```

---

## 📊 Metrics Explanation

### Reward Components

**Service Level (+10.0 × SL):**
- Measures customer satisfaction
- Range: [0, 1]
- Target: >0.95
- Higher is better

**Holding Cost (-0.1 × HC):**
- Cost of carrying inventory
- Units: dollars
- Lower is better

**Stockout Penalty (-5.0 × SP):**
- Lost sales cost
- Units: dollars
- Lower is better (0 is best)

**Excess Inventory (-0.05 × EI):**
- Inventory above target
- Units: units
- Lower is better

### Training Metrics

**Episode Reward:**
- Sum of rewards over episode
- Indicates overall performance
- Should increase over training

**Service Level:**
- Fraction of demand fulfilled
- Should remain >95%
- Slight decrease acceptable if costs improve

**Stockouts:**
- Total lost sales
- Should decrease over training
- Target: near zero

**Holding Costs:**
- Total inventory carrying cost
- Should decrease over training
- Balance with service level

---

## ✅ Success Criteria

**Training is successful if:**
1. ✅ Validation passes all tests
2. ✅ Training completes without errors
3. ✅ Episode rewards increase over time
4. ✅ PPO outperforms baseline in at least 2 metrics
5. ✅ Service level remains >90%

**Hackathon-ready if:**
1. ✅ Can demonstrate learning curve
2. ✅ Can show baseline comparison
3. ✅ Can explain HMARL architecture
4. ✅ Can discuss reconciliation-driven rewards

---

## 🎯 Hackathon Presentation Tips

### Key Points to Emphasize

1. **HMARL Architecture:**
   - Hierarchical: Store → Warehouse → Supplier
   - Multi-Agent: 4 autonomous agents
   - Reinforcement Learning: PPO with reconciliation rewards

2. **Reconciliation-Driven:**
   - Rewards from business metrics
   - No hand-crafted reward shaping
   - Realistic optimization objectives

3. **CTDE:**
   - Centralized training for efficiency
   - Decentralized execution for scalability
   - Parameter sharing for generalization

4. **Results:**
   - Show training plots
   - Show baseline comparison
   - Quantify improvements

### Demo Flow

1. **Show validation** (30 seconds)
2. **Explain architecture** (2 minutes)
3. **Show training plots** (1 minute)
4. **Show baseline comparison** (1 minute)
5. **Discuss extensibility** (1 minute)

---

**Ready to train?**

```bash
# Run complete pipeline
python training/validate_environment.py && \
python training/train_ppo_phase1.py && \
python training/compare_baseline_vs_ppo.py
```

---

*Training Guide - HMARL System*  
*Last Updated: January 28, 2026*
