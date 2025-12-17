#!/usr/bin/env python3
"""
Run TabPFN experiments on top 3 datasets: Splice, Ionosphere, Pol

This script automates running active learning experiments with TabPFN
on the datasets that show the strongest US (Uncertainty Sampling) advantage.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json

# Top 3 datasets based on variance analysis
DATASETS = {
    'splice': {
        'variance': 6.26,
        'us_advantage': 3.93,
        'description': 'Splice junction gene sequences'
    },
    'ionosphere': {
        'variance': 11.00,
        'us_advantage': 3.24,
        'description': 'Radar returns from ionosphere'
    },
    'pol': {
        'variance': 16.07,
        'us_advantage': 2.84,
        'description': 'Political data'
    }
}

# Query strategies to evaluate
QUERY_STRATEGIES = {
    'US': {
        'name': 'margin-zhan',
        'hs': 'google-zhan',
        'tool': 'google',
        'description': 'Uncertainty Sampling (Compatible)',
    },
    'US-NC': {
        'name': 'margin-nc',
        'hs': 'google-zhan',
        'tool': 'google',
        'description': 'Uncertainty Sampling (Non-compatible)',
    },
    'Random': {
        'name': 'random',
        'hs': 'random',
        'tool': 'google',
        'description': 'Random Sampling (Baseline)',
    },
    'QBC': {
        'name': 'qbc',
        'hs': 'google-zhan',
        'tool': 'google',
        'description': 'Query By Committee',
    },
    'BALD': {
        'name': 'margin-zhan',  # Use US as proxy for BALD if not available
        'hs': 'google-zhan',
        'tool': 'google',
        'description': 'Bayesian Active Learning by Disagreement',
    },
    'Core-Set': {
        'name': 'core-set',
        'hs': 'google-zhan',
        'tool': 'google',
        'description': 'Core-Set Selection',
    },
}

# Default experimental settings
DEFAULT_CONFIG = {
    'init_lbl_size': 20,      # Initial labeled pool size
    'tst_size': 0.4,          # Test set size (40%)
    'budget': 200,            # Total labeling budget
    'n_trials': 10,           # Number of independent trials
    'seed_start': 0,          # Starting seed for trials
    'exp_name': 'TabPFN',     # Experiment name
    'model': 'tabpfn',        # Model type
}


class ExperimentRunner:
    """Manages and executes TabPFN active learning experiments"""
    
    def __init__(self, config=None, output_dir='../results', log_file=None):
        """
        Initialize experiment runner
        
        Args:
            config: Dictionary with experimental configuration
            output_dir: Directory to save results
            log_file: Path to log file
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        if log_file:
            self.log_file = Path(log_file)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.log_file = self.output_dir / f'tabpfn_experiment_{timestamp}.log'
        
        self.results = []
        
    def log(self, message, level='INFO'):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {level}: {message}"
        print(log_msg)
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def run_single_experiment(self, dataset, strategy_name, seed):
        """
        Run a single experiment
        
        Args:
            dataset: Dataset name
            strategy_name: Query strategy name
            seed: Random seed
            
        Returns:
            success: Whether experiment succeeded
        """
        strategy = QUERY_STRATEGIES[strategy_name]
        
        cmd = [
            'python', 'main.py',
            '--data_set', dataset,
            '--tool', strategy['tool'],
            '--qs_name', strategy['name'],
            '--hs_name', strategy['hs'],
            '--gs_name', 'zhan',
            '--model', self.config['model'],
            '--seed', str(seed),
            '--n_trials', '1',
            '--init_lbl_size', str(self.config['init_lbl_size']),
            '--tst_size', str(self.config['tst_size']),
            '--budget', str(self.config['budget']),
            '--exp_name', self.config['exp_name'],
        ]
        
        self.log(f"Running: {strategy_name} on {dataset} (seed={seed})")
        self.log(f"Command: {' '.join(cmd)}", level='DEBUG')
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                self.log(f"✓ Success: {strategy_name} on {dataset} (seed={seed})")
                return True
            else:
                self.log(f"✗ Failed: {result.stderr}", level='ERROR')
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"✗ Timeout: {strategy_name} on {dataset} (seed={seed})", level='ERROR')
            return False
        except Exception as e:
            self.log(f"✗ Exception: {e}", level='ERROR')
            return False
    
    def run_all_experiments(self, datasets=None, strategies=None):
        """
        Run all experiments
        
        Args:
            datasets: List of dataset names (default: all top 3)
            strategies: List of strategy names (default: all)
        """
        datasets = datasets or list(DATASETS.keys())
        strategies = strategies or list(QUERY_STRATEGIES.keys())
        
        self.log("="*80)
        self.log("Starting TabPFN Active Learning Experiments")
        self.log("="*80)
        self.log(f"Datasets: {', '.join(datasets)}")
        self.log(f"Strategies: {', '.join(strategies)}")
        self.log(f"Trials per experiment: {self.config['n_trials']}")
        self.log(f"Total experiments: {len(datasets) * len(strategies) * self.config['n_trials']}")
        self.log("="*80)
        
        total = 0
        successful = 0
        
        for dataset in datasets:
            self.log(f"\n{'='*60}")
            self.log(f"Dataset: {dataset.upper()}")
            self.log(f"Variance: {DATASETS[dataset]['variance']:.2f}")
            self.log(f"US Advantage: +{DATASETS[dataset]['us_advantage']:.2f}%")
            self.log(f"{'='*60}\n")
            
            for strategy_name in strategies:
                for trial in range(self.config['n_trials']):
                    seed = self.config['seed_start'] + trial
                    total += 1
                    
                    success = self.run_single_experiment(dataset, strategy_name, seed)
                    
                    self.results.append({
                        'dataset': dataset,
                        'strategy': strategy_name,
                        'trial': trial,
                        'seed': seed,
                        'success': success,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    if success:
                        successful += 1
        
        # Print summary
        self.print_summary(total, successful)
        self.save_results()
    
    def print_summary(self, total, successful):
        """Print experiment summary"""
        self.log("\n" + "="*80)
        self.log("EXPERIMENT SUMMARY")
        self.log("="*80)
        
        failed = total - successful
        success_rate = (successful / total * 100) if total > 0 else 0
        
        self.log(f"Total experiments: {total}")
        self.log(f"Successful: {successful} ({success_rate:.1f}%)")
        self.log(f"Failed: {failed}")
        
        if failed > 0:
            self.log("\nFailed experiments:")
            for r in self.results:
                if not r['success']:
                    self.log(f"  - {r['dataset']}, {r['strategy']}, trial {r['trial']}")
        
        # Per-dataset summary
        self.log("\nPer-Dataset Success Rate:")
        for dataset in DATASETS:
            dataset_results = [r for r in self.results if r['dataset'] == dataset]
            if dataset_results:
                dataset_success = sum(1 for r in dataset_results if r['success'])
                dataset_total = len(dataset_results)
                rate = dataset_success / dataset_total * 100
                self.log(f"  {dataset}: {dataset_success}/{dataset_total} ({rate:.1f}%)")
    
    def save_results(self):
        """Save experiment results to JSON"""
        results_file = self.output_dir / f"tabpfn_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'config': self.config,
                'datasets': DATASETS,
                'strategies': QUERY_STRATEGIES,
                'results': self.results
            }, f, indent=2)
        
        self.log(f"\nResults saved to: {results_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run TabPFN active learning experiments on top 3 datasets'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help='Datasets to run experiments on'
    )
    
    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=list(QUERY_STRATEGIES.keys()),
        default=list(QUERY_STRATEGIES.keys()),
        help='Query strategies to evaluate'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=10,
        help='Number of trials per experiment (default: 10)'
    )
    
    parser.add_argument(
        '--budget',
        type=int,
        default=200,
        help='Labeling budget (default: 200)'
    )
    
    parser.add_argument(
        '--init_size',
        type=int,
        default=20,
        help='Initial labeled pool size (default: 20)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--quick_test',
        action='store_true',
        help='Run quick test with 1 trial on 1 dataset'
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'n_trials': 1 if args.quick_test else args.trials,
        'budget': args.budget,
        'init_lbl_size': args.init_size,
    }
    
    # Quick test mode
    if args.quick_test:
        print("="*80)
        print("QUICK TEST MODE")
        print("Running 1 trial on 'splice' dataset with 'US' strategy")
        print("="*80)
        args.datasets = ['splice']
        args.strategies = ['US']
    
    # Run experiments
    runner = ExperimentRunner(config=config, output_dir=args.output_dir)
    runner.run_all_experiments(datasets=args.datasets, strategies=args.strategies)


if __name__ == '__main__':
    main()
