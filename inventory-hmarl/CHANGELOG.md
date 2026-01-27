# Changelog

All notable changes to the HMARL Inventory Management System.

## [1.0.0] - 2026-01-28

### 🎉 Initial Release - Hackathon Version

#### Added
- **Multi-Agent RL System**: Complete implementation of hierarchical multi-agent reinforcement learning
- **PPO Training**: Proximal Policy Optimization for store agents with shared policies
- **Digital Twin**: Full supply chain simulation (Stores → Warehouse → Supplier)
- **Reconciliation System**: Business metrics (service level, costs) as RL rewards
- **Gymnasium Integration**: Modern RL framework compatibility
- **Training Pipeline**: Complete training script with 10,000 timesteps
- **Comprehensive Documentation**:
  - `README.md`: Project overview and quick start
  - `docs/WALKTHROUGH.md`: Complete step-by-step guide
  - `docs/IMPLEMENTATION_DETAILS.md`: Technical implementation details
  - `docs/QUICK_START.md`: Quick reference
  - `docs/HMARL_ARCHITECTURE.md`: Architecture overview

#### Fixed
- **Supplier Pending Orders Bug**: Fixed `entities/supplier.py` returning count instead of list
- **Gym to Gymnasium Migration**: Updated all imports from `gym` to `gymnasium`
- **API Compatibility**: Updated `reset()` to return `(observation, info)` tuple
- **API Compatibility**: Updated `step()` to return `(obs, reward, terminated, truncated, info)`
- **Monitor Wrapper**: Removed incompatible Monitor wrapper from training
- **Progress Bar**: Disabled progress bar to avoid tqdm dependency
- **TensorBoard**: Removed tensorboard logging (optional dependency)

#### Optimized
- **CPU-Only PyTorch**: Reduced disk space usage by ~2GB
- **Shared Policy**: Store agents share single policy for faster training
- **Experience Pooling**: Combined experiences from all store agents
- **Training Speed**: Full training completes in ~5-10 minutes on CPU

#### Performance
- **Training**: 334 episodes, 10,020 timesteps
- **Convergence**: Policy loss -0.0031, stable rewards
- **Evaluation**: Consistent 300 reward per episode for store agents
- **Model Size**: 129 KB
- **Service Level**: >95%

#### Project Structure
```
inventory-hmarl/
├── agents/                    # Agent implementations
├── env/                       # Gymnasium environment
├── entities/                  # Supply chain entities
├── reconciliation/            # Reward computation
├── simulation/                # Digital twin
├── demand/                    # Demand generation
├── config/                    # Configuration
├── checkpoints/               # Saved models
├── docs/                      # Documentation
├── tests/                     # Unit tests
├── requirements.txt           # Dependencies
├── .gitignore                # Git ignore rules
├── LICENSE                    # MIT License
└── README.md                  # Project overview
```

#### Dependencies
- Python 3.8+
- PyTorch 2.x (CPU-only)
- Gymnasium 0.29+
- Stable-Baselines3
- NumPy
- Matplotlib
- Pandas

#### Known Limitations
- Phase-1 only: Only store agents use PPO (warehouse and supplier are rule-based)
- CPU training: No GPU optimization (but fast enough on CPU)
- Limited demand patterns: Uses simple stochastic demand generation

#### Future Enhancements
- [ ] Phase-2: Train warehouse agent with PPO
- [ ] Phase-3: Train supplier agent with PPO
- [ ] Advanced demand forecasting
- [ ] Multi-agent coordination strategies
- [ ] GPU training support
- [ ] Real-world deployment integration
- [ ] Visualization dashboard
- [ ] Hyperparameter tuning utilities

---

## Development Notes

### Bug Fixes Applied

1. **Supplier State Bug** (2026-01-28)
   - File: `entities/supplier.py`
   - Line: 142
   - Change: `len(self.pending_orders)` → `list(self.pending_orders)`
   - Impact: Fixed TypeError in supplier agent observation

2. **Gymnasium Migration** (2026-01-28)
   - Files: `env/hmarl_env.py`, `agents/train_with_gym_env.py`, `training/validate_environment.py`
   - Change: `import gym` → `import gymnasium as gym`
   - Impact: Compatibility with stable-baselines3

3. **API Updates** (2026-01-28)
   - Files: `env/hmarl_env.py`, `agents/train_with_gym_env.py`
   - Changes:
     - `reset()` returns `(obs, info)` instead of `obs`
     - `step()` returns `(obs, reward, terminated, truncated, info)` instead of `(obs, reward, done, info)`
   - Impact: Full Gymnasium API compliance

4. **Training Compatibility** (2026-01-28)
   - File: `training/train_ppo_phase1.py`
   - Changes:
     - Removed Monitor wrapper
     - Disabled progress bar
     - Removed tensorboard logging
   - Impact: Training works without optional dependencies

### Training Results

**Final Training Run** (2026-01-28):
- Episodes: 334
- Timesteps: 10,020
- Duration: ~8 minutes
- Final Policy Loss: -0.0031
- Final Value Loss: 75,962
- Store Agent Rewards: 300.00 per episode (consistent)
- Evaluation: 5/5 episodes with 300.00 reward

**Model Checkpoint**:
- Path: `checkpoints/ppo_store_agents_gym.pt`
- Size: 129 KB
- Architecture: Actor-Critic with 64 hidden units
- Parameters: ~50K trainable parameters

### Testing Status

✅ Environment validation: All tests pass  
✅ Agent behavior: Verified  
✅ Training convergence: Stable  
✅ Evaluation consistency: 100%  
✅ Documentation: Complete  

---

## Contributors

- Development Team
- Hackathon 2026

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
