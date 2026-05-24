#!/usr/bin/env python3
"""
Feature Selection Tool
A simple tool to find optimal variable combinations using LightGBM evaluation.
"""

import argparse
import json
import sys
from itertools import combinations
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb


class FeatureSelector:
    """Main class for feature selection using LightGBM."""
    
    def __init__(self, dataset_path: str, target_column: str, 
                 mandatory_vars: List[str], min_vars: int, max_vars: int,
                 test_size: float = 0.2, random_seed: int = 42):
        """
        Initialize the feature selector.
        
        Args:
            dataset_path: Path to CSV dataset file
            target_column: Name of target variable column
            mandatory_vars: List of variables that must be included
            min_vars: Minimum number of variables in combinations
            max_vars: Maximum number of variables in combinations
            test_size: Fraction of data for test set (default: 0.2)
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.mandatory_vars = mandatory_vars
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.test_size = test_size
        self.random_seed = random_seed
        
        # LightGBM default parameters for regression
        self.lgbm_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_seed
        }
        
        # Load and split data
        self.X_train, self.X_test, self.y_train, self.y_test = self._load_and_split_data()
        
        # Available variables (excluding target and mandatory vars)
        self.available_vars = [col for col in self.X_train.columns 
                               if col not in self.mandatory_vars]
        
        # Cache for evaluation results
        self.eval_cache = {}
        
    def _load_and_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load CSV and split into train/test sets."""
        try:
            df = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            print(f"Error: Dataset file '{self.dataset_path}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)
        
        # Validate target column exists
        if self.target_column not in df.columns:
            print(f"Error: Target column '{self.target_column}' not found in dataset.")
            print(f"Available columns: {', '.join(df.columns)}")
            sys.exit(1)
        
        # Validate mandatory variables exist
        missing_vars = [var for var in self.mandatory_vars if var not in df.columns]
        if missing_vars:
            print(f"Error: Mandatory variables not found in dataset: {', '.join(missing_vars)}")
            print(f"Available columns: {', '.join(df.columns)}")
            sys.exit(1)
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Validate minimum variables
        if len(X.columns) < 2:
            print(f"Error: Dataset must have at least 2 feature columns (found {len(X.columns)}).")
            sys.exit(1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed
        )
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_combination(self, variables: List[str]) -> float:
        """
        Evaluate a variable combination using LightGBM and return R² score.
        
        Args:
            variables: List of variable names to use
            
        Returns:
            R² score on test set
        """
        # Use cache to avoid redundant computation
        var_key = frozenset(variables)
        if var_key in self.eval_cache:
            return self.eval_cache[var_key]
        
        # Select variables
        X_train_subset = self.X_train[variables]
        X_test_subset = self.X_test[variables]
        
        # Train LightGBM model
        train_data = lgb.Dataset(X_train_subset, label=self.y_train)
        model = lgb.train(
            self.lgbm_params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=0)]
        )
        
        # Predict and compute R²
        y_pred = model.predict(X_test_subset)
        r2 = r2_score(self.y_test, y_pred)
        
        # Cache result
        self.eval_cache[var_key] = r2
        
        return r2
    
    def exhaustive_search(self) -> Dict[str, Any]:
        """
        Perform exhaustive search over all valid variable combinations.
        
        Returns:
            Dictionary with best combination and results
        """
        print("\n=== Exhaustive Search ===")
        print(f"Mandatory variables: {self.mandatory_vars}")
        print(f"Variable range: {self.min_vars} to {self.max_vars}")
        print(f"Available variables to choose from: {len(self.available_vars)}")
        
        best_combination = None
        best_score = -float('inf')
        total_evaluated = 0
        
        # Generate all combinations
        for size in range(self.min_vars, self.max_vars + 1):
            # Number of additional variables needed
            n_additional = size - len(self.mandatory_vars)
            
            if n_additional < 0:
                continue
            
            if n_additional == 0:
                # Only mandatory variables
                combo_list = [self.mandatory_vars]
            else:
                # Generate combinations of additional variables
                combo_list = []
                for additional_vars in combinations(self.available_vars, n_additional):
                    combo = self.mandatory_vars + list(additional_vars)
                    combo_list.append(combo)
            
            # Evaluate each combination
            for combo in combo_list:
                total_evaluated += 1
                score = self.evaluate_combination(combo)
                
                if score > best_score:
                    best_score = score
                    best_combination = combo
                    print(f"[{total_evaluated}] New best! R² = {score:.4f}, Variables: {combo}")
                
                # Progress update every 50 evaluations
                if total_evaluated % 50 == 0:
                    print(f"[{total_evaluated}] Evaluated {total_evaluated} combinations, Best R² = {best_score:.4f}")
        
        print(f"\n✓ Exhaustive search complete!")
        print(f"Total combinations evaluated: {total_evaluated}")
        print(f"Best R² score: {best_score:.4f}")
        print(f"Best combination ({len(best_combination)} variables): {best_combination}")
        
        return {
            'strategy': 'exhaustive',
            'best_combination': best_combination,
            'best_score': best_score,
            'total_evaluated': total_evaluated
        }
    
    def forward_selection(self) -> Dict[str, Any]:
        """
        Perform forward selection starting from mandatory variables.
        
        Returns:
            Dictionary with best combination and results
        """
        print("\n=== Forward Selection ===")
        print(f"Starting with mandatory variables: {self.mandatory_vars}")
        print(f"Target: {self.min_vars} to {self.max_vars} variables")
        
        current_vars = self.mandatory_vars.copy()
        current_score = self.evaluate_combination(current_vars)
        total_evaluated = 1
        
        print(f"[{total_evaluated}] Initial R² = {current_score:.4f} with {len(current_vars)} variables")
        
        # Iteratively add variables
        while len(current_vars) < self.max_vars:
            best_var_to_add = None
            best_score_improvement = 0
            
            # Try adding each remaining variable
            for var in self.available_vars:
                if var in current_vars:
                    continue
                
                candidate_vars = current_vars + [var]
                total_evaluated += 1
                score = self.evaluate_combination(candidate_vars)
                improvement = score - current_score
                
                if improvement > best_score_improvement:
                    best_score_improvement = improvement
                    best_var_to_add = var
                    best_new_score = score
            
            # Stop if no improvement or reached min_vars with no improvement
            if best_var_to_add is None or (len(current_vars) >= self.min_vars and best_score_improvement <= 0):
                print(f"✓ No further improvement found. Stopping at {len(current_vars)} variables.")
                break
            
            # Add the best variable
            current_vars.append(best_var_to_add)
            current_score = best_new_score
            print(f"[{total_evaluated}] Added '{best_var_to_add}': R² = {current_score:.4f} (+{best_score_improvement:.4f})")
        
        print(f"\n✓ Forward selection complete!")
        print(f"Total combinations evaluated: {total_evaluated}")
        print(f"Best R² score: {current_score:.4f}")
        print(f"Best combination ({len(current_vars)} variables): {current_vars}")
        
        return {
            'strategy': 'forward',
            'best_combination': current_vars,
            'best_score': current_score,
            'total_evaluated': total_evaluated
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str = 'results.json'):
        """Save results to JSON file."""
        output_data = {
            'optimal_combination': results['best_combination'],
            'performance_metric': {
                'name': 'R²',
                'value': results['best_score']
            },
            'search_summary': {
                'strategy': results['strategy'],
                'combinations_evaluated': results['total_evaluated'],
                'total_variables_available': len(self.X_train.columns)
            },
            'configuration': {
                'mandatory_variables': self.mandatory_vars,
                'min_vars': self.min_vars,
                'max_vars': self.max_vars,
                'test_size': self.test_size,
                'random_seed': self.random_seed
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")


def main():
    """Main entry point for the feature selection tool."""
    parser = argparse.ArgumentParser(
        description='Feature Selection Tool - Find optimal variable combinations using LightGBM'
    )
    
    parser.add_argument('--dataset', required=True, help='Path to CSV dataset file')
    parser.add_argument('--target', required=True, help='Name of target variable column')
    parser.add_argument('--mandatory', nargs='+', default=[], help='List of mandatory variables')
    parser.add_argument('--min-vars', type=int, required=True, help='Minimum number of variables')
    parser.add_argument('--max-vars', type=int, required=True, help='Maximum number of variables')
    parser.add_argument('--strategy', choices=['exhaustive', 'forward'], required=True,
                        help='Search strategy to use')
    parser.add_argument('--test-size', type=float, default=0.2, 
                        help='Test set size as fraction (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output', default='results.json', help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min_vars < 1:
        print("Error: --min-vars must be at least 1")
        sys.exit(1)
    
    if args.max_vars > 30:
        print("Error: --max-vars must not exceed 30")
        sys.exit(1)
    
    if args.min_vars > args.max_vars:
        print("Error: --min-vars must not exceed --max-vars")
        sys.exit(1)
    
    if len(args.mandatory) > args.max_vars:
        print(f"Error: Number of mandatory variables ({len(args.mandatory)}) exceeds --max-vars ({args.max_vars})")
        sys.exit(1)
    
    if not (0.1 <= args.test_size <= 0.5):
        print("Error: --test-size must be between 0.1 and 0.5")
        sys.exit(1)
    
    # Create feature selector
    selector = FeatureSelector(
        dataset_path=args.dataset,
        target_column=args.target,
        mandatory_vars=args.mandatory,
        min_vars=args.min_vars,
        max_vars=args.max_vars,
        test_size=args.test_size,
        random_seed=args.seed
    )
    
    # Run selected strategy
    if args.strategy == 'exhaustive':
        results = selector.exhaustive_search()
    elif args.strategy == 'forward':
        results = selector.forward_selection()
    else:
        print(f"Error: Unknown strategy '{args.strategy}'")
        sys.exit(1)
    
    # Save results
    selector.save_results(results, args.output)
    
    print("\n" + "="*60)
    print("Feature selection complete!")
    print("="*60)


if __name__ == '__main__':
    main()
