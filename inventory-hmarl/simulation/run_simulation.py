"""
run_simulation.py

Main entry point for running digital twin simulations.

Usage:
    python simulation/run_simulation.py
    python simulation/run_simulation.py --days 180 --scenario spike
    python simulation/run_simulation.py --days 90 --scenario normal --seed 123
"""

import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.digital_twin import DigitalTwin
from scenarios.scenarios import get_scenario, SCENARIOS
import config.simulation_config as config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run retail supply chain digital twin simulation'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=None,
        help=f'Number of days to simulate (default: {config.SIMULATION_DAYS})'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='normal',
        choices=list(SCENARIOS.keys()),
        help='Scenario to run (default: normal)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help=f'Random seed (default: {config.RANDOM_SEED})'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output (only show final summary)'
    )
    
    return parser.parse_args()


def main():
    """Main simulation runner."""
    args = parse_args()
    
    # Build configuration
    sim_config = {
        'SIMULATION_DAYS': args.days if args.days else config.SIMULATION_DAYS,
        'WARMUP_DAYS': config.WARMUP_DAYS,
        'RANDOM_SEED': args.seed if args.seed else config.RANDOM_SEED,
        'SKU_CONFIG': config.SKU_CONFIG,
        'STORE_CONFIG': config.STORE_CONFIG,
        'WAREHOUSE_CONFIG': config.WAREHOUSE_CONFIG,
        'SUPPLIER_CONFIG': config.SUPPLIER_CONFIG
    }
    
    # Print simulation info
    if not args.quiet:
        print("\n" + "="*60)
        print("RETAIL SUPPLY CHAIN DIGITAL TWIN SIMULATION")
        print("="*60)
        print(f"Scenario: {args.scenario}")
        print(f"Simulation Days: {sim_config['SIMULATION_DAYS']}")
        print(f"Warmup Days: {sim_config['WARMUP_DAYS']}")
        print(f"Random Seed: {sim_config['RANDOM_SEED']}")
        print(f"Stores: {len(config.STORE_CONFIG)}")
        print(f"Warehouses: {len(config.WAREHOUSE_CONFIG)}")
        print(f"Suppliers: {len(config.SUPPLIER_CONFIG)}")
        print(f"SKUs: {len(config.SKU_CONFIG)}")
        print("="*60)
        print()
    
    # Initialize digital twin
    env = DigitalTwin(sim_config)
    
    # Load and apply scenario
    scenario = get_scenario(args.scenario)
    
    if not args.quiet:
        print(f"Applying scenario: {scenario.name}")
        print(f"Description: {scenario.description}")
        print()
    
    # Apply scenario modifications
    env.reset()  # Reset first
    scenario.apply_to_environment(env)
    
    # Run simulation
    if not args.quiet:
        print("Starting simulation...")
        print()
    
    metrics = env.run_simulation(
        num_days=sim_config['SIMULATION_DAYS'],
        verbose=args.verbose
    )
    
    # Print final summary
    if not args.quiet:
        env.metrics_tracker.print_summary()
    
    # Return metrics for programmatic use
    return metrics


if __name__ == '__main__':
    try:
        metrics = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
