#!/usr/bin/env python3
"""
Script to run CAKE attention analysis and visualization.

This script demonstrates how to:
1. Run inference with attention score collection
2. Analyze attention patterns across layers and steps
3. Visualize attention interactions between tokens
4. Generate comprehensive reports

Usage:
    python run_attention_analysis.py --model llama3.1-8b-128k --compress --save_attention --max_samples 3
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from experiments.LongBench.pred_cake_attention_analysis import main as run_attention_collection
from experiments.LongBench.visualize_attention import AttentionVisualizer
from experiments.LongBench.attention_analysis_utils import AttentionAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Run CAKE attention analysis and visualization")
    
    # Model and inference arguments
    parser.add_argument('--model', type=str, default="llama3.1-8b-128k", 
                        choices=["llama2-7b-chat-4k", "llama2-13b-chat-4k", "llama3.1-8b-128k", 
                                "mistral-0.3-7b-32k", "qwen2.5-7b-instruct"],
                        help="Model to use for analysis")
    parser.add_argument('--compress', action='store_true', 
                        help="Enable CAKE compression")
    parser.add_argument('--cache_size', type=int, default=1024,
                        help="Cache size for CAKE compression")
    parser.add_argument('--window_size', type=int, default=32,
                        help="Window size for CAKE compression")
    parser.add_argument('--tau1', type=float, default=1.0,
                        help="Tau1 parameter for CAKE")
    parser.add_argument('--tau2', type=float, default=1.0,
                        help="Tau2 parameter for CAKE")
    parser.add_argument('--gamma', type=float, default=200.0,
                        help="Gamma parameter for CAKE")
    
    # Analysis arguments
    parser.add_argument('--max_samples', type=int, default=3,
                        help="Maximum number of samples to analyze")
    parser.add_argument('--dataset', type=str, default="narrativeqa",
                        help="Dataset to analyze")
    parser.add_argument('--device', type=int, default=0,
                        help="GPU device to use")
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default="./attention_analysis",
                        help="Output directory for analysis results")
    parser.add_argument('--skip_collection', action='store_true',
                        help="Skip data collection and only run analysis/visualization")
    parser.add_argument('--skip_analysis', action='store_true',
                        help="Skip analysis and only run visualization")
    parser.add_argument('--skip_visualization', action='store_true',
                        help="Skip visualization and only run analysis")
    
    return parser.parse_args()

def run_attention_data_collection(args):
    """Run the attention data collection"""
    print("=" * 60)
    print("STEP 1: Collecting Attention Data")
    print("=" * 60)
    
    # Set up arguments for the attention collection script
    collection_args = [
        '--model', args.model,
        '--max_samples', str(args.max_samples),
        '--dataset', args.dataset,
        '--device', str(args.device),
        '--output_dir', args.output_dir,
        '--save_attention'
    ]
    
    if args.compress:
        collection_args.extend([
            '--compress',
            '--cache_size', str(args.cache_size),
            '--window_size', str(args.window_size),
            '--tau1', str(args.tau1),
            '--tau2', str(args.tau2),
            '--gamma', str(args.gamma)
        ])
    
    # Run the collection script
    sys.argv = ['pred_cake_attention_analysis.py'] + collection_args
    run_attention_collection()

def run_attention_analysis(args):
    """Run the attention analysis"""
    print("=" * 60)
    print("STEP 2: Analyzing Attention Patterns")
    print("=" * 60)
    
    analyzer = AttentionAnalyzer(args.output_dir)
    
    for sample_idx in range(args.max_samples):
        try:
            print(f"Analyzing sample {sample_idx}...")
            report = analyzer.generate_analysis_report(sample_idx, f"{args.output_dir}/reports")
            print(f"Analysis completed for sample {sample_idx}")
        except FileNotFoundError:
            print(f"No data found for sample {sample_idx}, skipping...")
            continue

def run_attention_visualization(args):
    """Run the attention visualization"""
    print("=" * 60)
    print("STEP 3: Generating Visualizations")
    print("=" * 60)
    
    visualizer = AttentionVisualizer(args.output_dir)
    
    for sample_idx in range(args.max_samples):
        try:
            print(f"Generating visualizations for sample {sample_idx}...")
            visualizer.generate_comprehensive_report(sample_idx, f"{args.output_dir}/plots")
            print(f"Visualizations completed for sample {sample_idx}")
        except FileNotFoundError:
            print(f"No data found for sample {sample_idx}, skipping...")
            continue

def main():
    args = parse_args()
    
    print("CAKE Attention Analysis Pipeline")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Compression: {'Enabled' if args.compress else 'Disabled'}")
    print(f"Max samples: {args.max_samples}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Collect attention data
    if not args.skip_collection:
        run_attention_data_collection(args)
    else:
        print("Skipping data collection...")
    
    # Step 2: Analyze attention patterns
    if not args.skip_analysis:
        run_attention_analysis(args)
    else:
        print("Skipping analysis...")
    
    # Step 3: Generate visualizations
    if not args.skip_visualization:
        run_attention_visualization(args)
    else:
        print("Skipping visualization...")
    
    print("=" * 60)
    print("Analysis pipeline completed!")
    print(f"Results saved in: {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
