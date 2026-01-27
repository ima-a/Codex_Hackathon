"""
run_complete_pipeline.py

Master script to run the complete PPO training pipeline.

Executes:
1. Environment validation
2. PPO training
3. Baseline comparison
4. Results summary

One-command execution for hackathon demo.
"""

import sys
import os
import subprocess
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_script(script_path: str, description: str) -> bool:
    """
    Run a Python script and return success status.
    
    Args:
        script_path: Path to script
        description: Description for logging
    
    Returns:
        True if successful
    """
    print_header(description)
    print(f"Running: {script_path}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.abspath(__file__)) + "/..",
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully")
            return True
        else:
            print(f"\n❌ {description} failed with code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ {description} failed with exception: {e}")
        return False


def main():
    """Run complete training pipeline."""
    start_time = datetime.now()
    
    print_header("HMARL PPO TRAINING PIPELINE")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nThis pipeline will:")
    print(f"  1. Validate the environment")
    print(f"  2. Train PPO agents")
    print(f"  3. Compare against baseline")
    print(f"  4. Generate results summary")
    
    # Step 1: Validation
    validation_script = "training/validate_environment.py"
    if not run_script(validation_script, "STEP 1: Environment Validation"):
        print("\n❌ PIPELINE FAILED: Validation did not pass")
        print("Fix validation errors before proceeding to training.")
        return 1
    
    # Step 2: Training
    training_script = "training/train_ppo_phase1.py"
    if not run_script(training_script, "STEP 2: PPO Training"):
        print("\n❌ PIPELINE FAILED: Training did not complete")
        return 1
    
    # Step 3: Comparison
    comparison_script = "training/compare_baseline_vs_ppo.py"
    if not run_script(comparison_script, "STEP 3: Baseline Comparison"):
        print("\n❌ PIPELINE FAILED: Comparison did not complete")
        return 1
    
    # Success!
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_header("PIPELINE COMPLETE")
    print(f"✅ All steps completed successfully!")
    print(f"\nTotal duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"\nGenerated outputs:")
    print(f"  📊 Training plots:     training_outputs/phase1/training_plots.png")
    print(f"  📈 Comparison plots:   comparison_outputs/baseline_vs_ppo_comparison.png")
    print(f"  💾 Trained model:      training_outputs/phase1/ppo_store_agents_phase1.zip")
    print(f"  📄 Training metrics:   training_outputs/phase1/training_metrics.json")
    print(f"  📄 Comparison summary: comparison_outputs/comparison_summary.json")
    
    print(f"\n🎉 Your HMARL system is trained and ready for demo!")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
