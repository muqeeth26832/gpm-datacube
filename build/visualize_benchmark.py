#!/usr/bin/env python3
"""
Visualize benchmark results comparing Datacube vs SimpleCube vs Parallel implementations.
Generates bar charts and speedup comparisons for performance analysis.
"""

import csv
import sys
import matplotlib.pyplot as plt
import numpy as np


def load_benchmark_data(filename: str) -> dict:
    """Load benchmark results from CSV file."""
    data = {
        'operation': [],
        'datacube': [],
        'simplecube': [],
        'parallel': [],
        'speedup_simple': [],
        'speedup_parallel': []
    }
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['operation'].append(row['operation'])
            data['datacube'].append(float(row['datacube_time']))
            data['simplecube'].append(float(row['simplecube_time']))
            data['parallel'].append(float(row['parallel_time']))
            data['speedup_simple'].append(float(row['speedup_simplecube']))
            data['speedup_parallel'].append(float(row['speedup_parallel']))
    
    return data


def plot_comparison_bar(data: dict, output: str | None = None) -> None:
    """Create a grouped bar chart comparing all three implementations."""
    operations = data['operation']
    datacube = data['datacube']
    simplecube = data['simplecube']
    parallel = data['parallel']
    
    x = np.arange(len(operations))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - width, datacube, width, label='Datacube (flat vector)', 
                   color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x, simplecube, width, label='SimpleCube (sequential)', 
                   color='#3498db', alpha=0.8)
    bars3 = ax.bar(x + width, parallel, width, label='SimpleCube (parallel)', 
                   color='#2ecc71', alpha=0.8)
    
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_xlabel('Operation', fontsize=12)
    ax.set_title('Performance Comparison: Datacube vs SimpleCube vs Parallel', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(operations, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars3:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved bar chart to {output}")
    else:
        plt.show()


def plot_speedup_comparison(data: dict, output: str | None = None) -> None:
    """Create a grouped bar chart showing speedup for both SimpleCube and Parallel."""
    operations = data['operation']
    speedup_simple = data['speedup_simple']
    speedup_parallel = data['speedup_parallel']
    
    x = np.arange(len(operations))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, speedup_simple, width, label='SimpleCube Speedup', 
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, speedup_parallel, width, label='Parallel Speedup', 
                   color='#2ecc71', alpha=0.8)
    
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Break-even (1x)')
    
    ax.set_ylabel('Speedup (Datacube / Implementation)', fontsize=12)
    ax.set_xlabel('Operation', fontsize=12)
    ax.set_title('Speedup Comparison (relative to Datacube)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(operations, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, s in zip(bars1, speedup_simple):
        height = bar.get_height()
        ax.annotate(f'{s:.2f}x',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    for bar, s in zip(bars2, speedup_parallel):
        height = bar.get_height()
        ax.annotate(f'{s:.2f}x',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved speedup chart to {output}")
    else:
        plt.show()


def plot_combined(data: dict, output: str | None = None) -> None:
    """Create a combined visualization with time comparison and speedup."""
    operations = data['operation']
    datacube = data['datacube']
    simplecube = data['simplecube']
    parallel = data['parallel']
    speedup_simple = data['speedup_simple']
    speedup_parallel = data['speedup_parallel']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left: Time comparison
    x = np.arange(len(operations))
    width = 0.25
    
    bars1 = ax1.bar(x - width, datacube, width, label='Datacube', 
                    color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x, simplecube, width, label='SimpleCube (seq)', 
                    color='#3498db', alpha=0.8)
    bars3 = ax1.bar(x + width, parallel, width, label='SimpleCube (parallel)', 
                    color='#2ecc71', alpha=0.8)
    
    ax1.set_ylabel('Time (seconds)', fontsize=11)
    ax1.set_xlabel('Operation', fontsize=11)
    ax1.set_title('Execution Time Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(operations, rotation=15, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Right: Speedup comparison
    x2 = np.arange(len(operations))
    width2 = 0.35
    
    colors1 = ['#3498db' if s >= 1 else '#e67e22' for s in speedup_simple]
    colors2 = ['#2ecc71' if s >= 1 else '#e67e22' for s in speedup_parallel]
    
    bars_s1 = ax2.bar(x2 - width2/2, speedup_simple, width2, color=colors1, 
                      alpha=0.8, edgecolor='black', linewidth=1.2, label='SimpleCube')
    bars_s2 = ax2.bar(x2 + width2/2, speedup_parallel, width2, color=colors2, 
                      alpha=0.8, edgecolor='black', linewidth=1.2, label='Parallel')
    
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Break-even (1x)')
    
    ax2.set_ylabel('Speedup (Datacube / Implementation)', fontsize=11)
    ax2.set_xlabel('Operation', fontsize=11)
    ax2.set_title('Speedup Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(operations, rotation=15, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on speedup chart
    for bar, s in zip(bars_s1, speedup_simple):
        height = bar.get_height()
        ax2.annotate(f'{s:.2f}x',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    
    for bar, s in zip(bars_s2, speedup_parallel):
        height = bar.get_height()
        ax2.annotate(f'{s:.2f}x',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Datacube vs SimpleCube vs Parallel - Performance Benchmark', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved combined chart to {output}")
    else:
        plt.show()


def print_summary(data: dict) -> None:
    """Print a text summary of the benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY: Datacube vs SimpleCube vs Parallel")
    print("=" * 80)
    print(f"{'Operation':<20} {'Datacube':<12} {'SimpleCube':<12} {'Parallel':<12} {'Speedup(SC)':<12} {'Speedup(P)':<10}")
    print("-" * 80)
    
    for i, op in enumerate(data['operation']):
        print(f"{op:<20} {data['datacube'][i]:<12.6f} {data['simplecube'][i]:<12.6f} "
              f"{data['parallel'][i]:<12.6f} {data['speedup_simple'][i]:<12.2f}x "
              f"{data['speedup_parallel'][i]:<10.2f}x")
    
    print("-" * 80)
    avg_speedup_sc = sum(data['speedup_simple']) / len(data['speedup_simple'])
    avg_speedup_p = sum(data['speedup_parallel']) / len(data['speedup_parallel'])
    print(f"{'Average Speedup:':<20} {'':<12} {'':<12} {'':<12} {avg_speedup_sc:<12.2f}x {avg_speedup_p:<10.2f}x")
    print("=" * 80 + "\n")


def main():
    filename = "benchmark_results.csv"
    output_combined = "benchmark_comparison.png"
    output_bar = "benchmark_time_comparison.png"
    output_speedup = "benchmark_speedup.png"
    
    # Parse arguments
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        output_combined = sys.argv[2]
    
    try:
        data = load_benchmark_data(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        print("Run the benchmark first: ./build/gpmcube (select option 3)")
        sys.exit(1)
    
    print_summary(data)
    
    # Generate plots
    plot_combined(data, output_combined)
    plot_comparison_bar(data, output_bar)
    plot_speedup_comparison(data, output_speedup)
    
    print("\nVisualization complete!")
    print(f"  - {output_combined} (combined view)")
    print(f"  - {output_bar} (time comparison)")
    print(f"  - {output_speedup} (speedup comparison)")


if __name__ == "__main__":
    main()
