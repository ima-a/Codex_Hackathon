# 🚀 Quick Setup & Run Guide

## Step 1: Install Dependencies

```bash
# Create virtual environment (already done)
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install gym numpy matplotlib stable-baselines3 torch
```

## Step 2: Run Validation (MANDATORY)

```bash
# Make sure you're in the project root
cd /home/Ima/work/hackathon/codex/inventory-hmarl

# Run validation
python training/validate_environment.py
```

**Expected Output:**
```
✅ PASS: reset() validation successful
✅ PASS: step() validation successful
✅ PASS: reconciliation validation successful
✅ PASS: action space validation successful

✅ ALL VALIDATIONS PASSED
Environment is ready for PPO training!
```

**If validation fails:** Fix errors before proceeding!

## Step 3: Run Complete Training Pipeline

### Option A: One-Command (Recommended)
```bash
python training/run_complete_pipeline.py
```

This will automatically run:
1. Validation
2. PPO Training
3. Baseline Comparison
4. Generate all plots and summaries

**Duration:** ~5-10 minutes

### Option B: Step-by-Step
```bash
# Step 1: Validate
python training/validate_environment.py

# Step 2: Train PPO
python training/train_ppo_phase1.py

# Step 3: Compare vs Baseline
python training/compare_baseline_vs_ppo.py
```

## Step 4: View Results

After training completes, check these files:

```bash
# Training plots
open training_outputs/phase1/training_plots.png

# Comparison plots
open comparison_outputs/baseline_vs_ppo_comparison.png

# Trained model
ls training_outputs/phase1/ppo_store_agents_phase1.zip

# Metrics
cat training_outputs/phase1/training_metrics.json
cat comparison_outputs/comparison_summary.json
```

## Troubleshooting

### Issue: Import errors
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Validation fails
```bash
# Check if all files are present
ls env/hmarl_env.py
ls agents/store_agent.py
ls config/simulation_config.py
```

### Issue: Training is slow
- Reduce timesteps in `training/train_ppo_phase1.py`
- Change `total_timesteps=10000` to `total_timesteps=5000`

## Quick Commands Reference

```bash
# Full pipeline (recommended for first run)
python training/run_complete_pipeline.py

# Just validation
python training/validate_environment.py

# Just training
python training/train_ppo_phase1.py

# Just comparison
python training/compare_baseline_vs_ppo.py
```

## What You'll Get

After successful run:

✅ **Trained PPO model** - `training_outputs/phase1/ppo_store_agents_phase1.zip`  
✅ **Training plots** - Shows learning progression  
✅ **Comparison plots** - Baseline vs PPO performance  
✅ **Metrics files** - JSON data for analysis  
✅ **TensorBoard logs** - For detailed training analysis  

## Next Steps

1. Review training plots to see learning curve
2. Check comparison plots to see improvements
3. Prepare demo presentation
4. (Optional) Extend to Phase-2 (warehouse training)

---

**Ready to run?**

```bash
source venv/bin/activate
python training/run_complete_pipeline.py
```

🎉 **Good luck with your hackathon!**
