# 🎓 PPO Training for HMARL System

**Phase-1: Store Agent Training with Centralized PPO**

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run complete pipeline (recommended)
python run_complete_pipeline.py
```

That's it! The pipeline will:
1. ✅ Validate environment
2. ✅ Train PPO agents
3. ✅ Compare vs baseline
4. ✅ Generate plots and summaries

---

## 📁 Files in This Directory

| File | Purpose | Output |
|------|---------|--------|
| `validate_environment.py` | Pre-training validation | Console report |
| `train_ppo_phase1.py` | PPO training script | Model + plots |
| `compare_baseline_vs_ppo.py` | Baseline comparison | Comparison plots |
| `run_complete_pipeline.py` | Master runner | All outputs |
| `TRAINING_GUIDE.md` | Full documentation | - |
| `PPO_TRAINING_SUMMARY.md` | Complete summary | - |
| `README.md` | This file | - |

---

## 📊 Expected Outputs

After running the pipeline:

```
training_outputs/
└── phase1/
    ├── ppo_store_agents_phase1.zip    # ← Trained model
    ├── training_plots.png             # ← 4-panel training viz
    ├── training_metrics.json          # ← Raw metrics
    └── tensorboard/                   # ← TensorBoard logs

comparison_outputs/
├── baseline_vs_ppo_comparison.png     # ← Comparison plots
└── comparison_summary.json            # ← Statistical summary
```

---

## 🎯 Training Scope

**Phase-1 (Current):**
- ✅ Train 2 Store agents
- ✅ Shared PPO policy
- ✅ CTDE architecture
- ⏸️ Warehouse: rule-based
- ⏸️ Supplier: rule-based

**Phase-2 (Future):**
- Freeze store agents
- Train warehouse agent
- Optional joint fine-tuning

**Phase-3 (Future):**
- Freeze all downstream
- Train supplier agent
- Full system optimization

---

## 📖 Documentation

- **Quick Start:** This file
- **Complete Guide:** `TRAINING_GUIDE.md`
- **Full Summary:** `PPO_TRAINING_SUMMARY.md`

---

## ⚡ Individual Scripts

### 1. Validation Only
```bash
python validate_environment.py
```

### 2. Training Only
```bash
python train_ppo_phase1.py
```

### 3. Comparison Only
```bash
python compare_baseline_vs_ppo.py
```

---

## 🐛 Troubleshooting

**Issue:** Validation fails
```bash
# Check environment setup
cd ..
python -c "from env.hmarl_env import HMARLEnvironment; print('OK')"
```

**Issue:** Import errors
```bash
# Ensure you're in project root
cd /home/Ima/work/hackathon/codex/inventory-hmarl
python training/run_complete_pipeline.py
```

**Issue:** Slow training
```bash
# Reduce timesteps in train_ppo_phase1.py
# Change: total_timesteps=10000 → total_timesteps=5000
```

---

## ✅ Success Criteria

Training is successful if:
- ✅ Validation passes all 4 tests
- ✅ Training completes without errors
- ✅ Episode rewards increase
- ✅ PPO outperforms baseline
- ✅ Plots are generated

---

## 🎉 Ready to Train!

```bash
python run_complete_pipeline.py
```

---

*Training Directory - HMARL System*
