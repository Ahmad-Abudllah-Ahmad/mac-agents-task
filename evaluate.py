import json
import numpy as np
from pathlib import Path
import sys
import os

# Ensure metrics modules can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metrics.valid_syntax_rate import evaluate_syntax_rate
from metrics.best_iou import evaluate_codes

def run_evaluation():
    print("Loading predictions...")
    with open("results/ground_truth.json", "r") as f:
        gt_codes = json.load(f)
        
    with open("results/baseline_predictions.json", "r") as f:
        baseline_codes = json.load(f)
        
    with open("results/enhanced_predictions.json", "r") as f:
        enhanced_codes = json.load(f)
        
    # Keep only intersecting keys in case of any generation failures
    keys = list(set(gt_codes.keys()) & set(baseline_codes.keys()) & set(enhanced_codes.keys()))
    
    print(f"\nEvaluating Baseline Model on {len(keys)} samples...")
    print("-" * 50)
    baseline_syntax = evaluate_syntax_rate(baseline_codes, verbose=False)["vsr"]
    print("Running IOU Evaluation (Baseline)...")
    baseline_eval = evaluate_codes(gt_codes, baseline_codes)
    
    print(f"\nEvaluating Enhanced Model on {len(keys)} samples...")
    print("-" * 50)
    enhanced_syntax = evaluate_syntax_rate(enhanced_codes, verbose=False)["vsr"]
    print("Running IOU Evaluation (Enhanced)...")
    enhanced_eval = evaluate_codes(gt_codes, enhanced_codes)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"{'Metric':<20} | {'Baseline':<10} | {'Enhanced':<10} | {'Improvement':<10}")
    print("-" * 55)
    
    vsr_imp = enhanced_syntax - baseline_syntax
    iou_imp = enhanced_eval['iou_best'] - baseline_eval['iou_best']
    
    print(f"{'Valid Syntax Rate':<20} | {baseline_syntax:.1%}      | {enhanced_syntax:.1%}      | {vsr_imp:+.1%}")
    print(f"{'Best IOU':<20} | {baseline_eval['iou_best']:.3f}      | {enhanced_eval['iou_best']:.3f}      | {iou_imp:+.3f}")
    
    # Save results
    final_results = {
        "baseline": {
            "vsr": baseline_syntax,
            "iou_best": baseline_eval["iou_best"]
        },
        "enhanced": {
            "vsr": enhanced_syntax,
            "iou_best": enhanced_eval["iou_best"]
        },
        "improvements": {
            "vsr": vsr_imp,
            "iou_best": iou_imp
        }
    }
    with open("results/final_evaluation.json", "w") as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    run_evaluation()
