#!/usr/bin/env python3
"""
OLAP Benchmark Visualization - All Operations
Compares: Sequential vs std::thread vs OMP (1-loop, 2-loop, 3-loop, tile, cubed)
Across different cube sizes (LAT × LON)
All times in MICROSECONDS (µs)
"""

import csv
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# STYLE CONFIGURATION
# ============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 15

# Color palette - 7 distinct colors
COLORS = {
    'sequential': '#e74c3c',      # Red
    'std_thread': '#3498db',       # Blue
    'omp_1loop': '#9b59b6',        # Purple
    'omp_2loop': '#e67e22',        # Orange
    'omp_3loop': '#1abc9c',        # Teal
    'omp_tile': '#2ecc71',         # Green
    'omp_cubed': '#34495e'         # Dark Blue
}

LABELS = {
    'sequential': 'Sequential',
    'std_thread': 'std::thread',
    'omp_1loop': 'OMP (1-loop)',
    'omp_2loop': 'OMP (2-loop collapse)',
    'omp_3loop': 'OMP (3-loop collapse)',
    'omp_tile': 'OMP (tile)',
    'omp_cubed': 'OMP (cubed)'
}

# ============================================================================
# DATA LOADING
# ============================================================================
def load_size_sweep_csv(filename: str) -> dict:
    """Load size sweep CSV file."""
    data = {
        'sizes': [],
        'sequential': [],
        'std_thread': [],
        'omp_1loop': [],
        'omp_2loop': [],
        'omp_3loop': [],
        'omp_tile': [],
        'omp_cubed': []
    }
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['sizes'].append(int(row['size']))
            data['sequential'].append(float(row['sequential']))
            data['std_thread'].append(float(row['std_thread']))
            data['omp_1loop'].append(float(row['omp_1loop']))
            data['omp_2loop'].append(float(row['omp_2loop']))
            data['omp_3loop'].append(float(row['omp_3loop']))
            data['omp_tile'].append(float(row['omp_tile']))
            data['omp_cubed'].append(float(row['omp_cubed']))
    
    return data


def load_all_benchmarks() -> dict:
    """Load all benchmark files."""
    files = {
        'slice': 'benchmark_slice_size_sweep.csv',
        'dice': 'benchmark_dice_size_sweep.csv',
        'rollup_mean': 'benchmark_rollup_mean_size_sweep.csv',
        'global_mean': 'benchmark_global_mean_size_sweep.csv',
        'region_mean': 'benchmark_region_mean_size_sweep.csv'
    }
    
    data = {}
    for name, filename in files.items():
        if os.path.exists(filename):
            data[name] = load_size_sweep_csv(filename)
            print(f"Loaded: {filename}")
        else:
            print(f"Warning: {filename} not found")
    
    return data


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_operation_comparison(data: dict, operation: str, output: str | None = None) -> None:
    """Plot comparison for a single operation across sizes."""
    sizes = data['sizes']
    x = np.arange(len(sizes))
    width = 0.11
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    methods = [
        ('sequential', data['sequential']),
        ('std_thread', data['std_thread']),
        ('omp_1loop', data['omp_1loop']),
        ('omp_2loop', data['omp_2loop']),
        ('omp_3loop', data['omp_3loop']),
        ('omp_tile', data['omp_tile']),
        ('omp_cubed', data['omp_cubed'])
    ]
    
    # Plot bars
    bars = []
    for i, (method, times) in enumerate(methods):
        if all(t == 0 for t in times):
            continue
        offset = (i - len(methods)/2 + 0.5) * width
        bar = ax.bar(x + offset, times, width, label=LABELS.get(method, method),
                     color=COLORS.get(method, '#999999'), alpha=0.85, 
                     edgecolor='black', linewidth=0.8)
        bars.append(bar)
    
    ax.set_xlabel('Cube Size (LAT × LON)', fontsize=12)
    ax.set_ylabel('Time (µs)', fontsize=12)
    ax.set_title(f'{operation.replace("_", " ").title()}: Parallelization Strategy Comparison\n' + 
                 f'({sizes[0]}×{sizes[0]} to {sizes[-1]}×{sizes[-1]})',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}×{s}' for s in sizes], rotation=25, ha='right')
    ax.legend(loc='upper left', ncol=4, fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_all_operations_grid(data: dict, output: str | None = None) -> None:
    """Plot all operations in a 2x3 grid."""
    operations = list(data.keys())
    n_ops = len(operations)
    
    cols = 3
    rows = (n_ops + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    titles = {
        'slice': 'Slice Time',
        'dice': 'Dice Time',
        'rollup_mean': 'Rollup Time Mean',
        'global_mean': 'Global Mean',
        'region_mean': 'Region Mean'
    }
    
    for idx, op in enumerate(operations):
        ax = axes[idx]
        sizes = data[op]['sizes']
        x = np.arange(len(sizes))
        width = 0.11
        
        methods = [
            ('sequential', data[op]['sequential']),
            ('std_thread', data[op]['std_thread']),
            ('omp_1loop', data[op]['omp_1loop']),
            ('omp_2loop', data[op]['omp_2loop']),
            ('omp_3loop', data[op]['omp_3loop']),
            ('omp_tile', data[op]['omp_tile']),
            ('omp_cubed', data[op]['omp_cubed'])
        ]
        
        for i, (method, times) in enumerate(methods):
            if all(t == 0 for t in times):
                continue
            offset = (i - len(methods)/2 + 0.5) * width
            ax.bar(x + offset, times, width, label=LABELS.get(method, method),
                   color=COLORS.get(method, '#999999'), alpha=0.85,
                   edgecolor='black', linewidth=0.8)
        
        ax.set_xlabel('Cube Size (LAT × LON)', fontsize=11)
        ax.set_ylabel('Time (µs)', fontsize=11)
        ax.set_title(titles.get(op, op), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}×{s}' for s in sizes], rotation=20, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_yscale('log')
    
    # Hide unused subplots
    for idx in range(len(operations), len(axes)):
        axes[idx].axis('off')
    
    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10, 
               bbox_to_anchor=(0.5, 1.02))
    
    plt.suptitle('OLAP Operations: Parallelization Strategy Benchmark\n' + 
                 '(Sequential vs std::thread vs OpenMP Variants)',
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_method_comparison_across_ops(data: dict, output: str | None = None) -> None:
    """Plot each method's performance across all operations."""
    operations = list(data.keys())
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    methods = ['sequential', 'std_thread', 'omp_1loop', 'omp_2loop', 'omp_3loop', 'omp_tile', 'omp_cubed']
    method_titles = ['Sequential', 'std::thread', 'OMP 1-loop', 'OMP 2-loop', 'OMP 3-loop', 'OMP Tile', 'OMP Cubed']
    
    for idx, (method, title) in enumerate(zip(methods, method_titles)):
        ax = axes[idx]
        
        for op in operations:
            if op not in data:
                continue
            sizes = data[op]['sizes']
            times = data[op][method]
            if all(t == 0 for t in times):
                continue
            ax.plot(sizes, times, 'o-', linewidth=2, markersize=8,
                    label=op.replace('_', ' ').title(), markerfacecolor='white', markeredgewidth=1.5)
        
        ax.set_xlabel('Cube Size (LAT × LON)', fontsize=11)
        ax.set_ylabel('Time (µs)', fontsize=11)
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
    
    # Hide last subplot if not needed
    if len(methods) < 8:
        axes[-1].axis('off')
    
    plt.suptitle('Performance by Method: All Operations\n' + 
                 '(Log-log scale, lower is better)',
                 fontsize=15, fontweight='bold', y=1.005)
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_speedup_heatmap(data: dict, output: str | None = None) -> None:
    """Plot speedup heatmap comparing all methods to sequential."""
    operations = list(data.keys())
    methods = ['std_thread', 'omp_1loop', 'omp_2loop', 'omp_3loop', 'omp_tile', 'omp_cubed']
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Speedup at largest size
    ax = axes[0]
    
    largest_idx = -1
    speedup_data = []
    y_labels = []
    
    for op in operations:
        if op not in data:
            continue
        y_labels.append(op.replace('_', '\n').title())
        seq = data[op]['sequential'][largest_idx]
        row = []
        for method in methods:
            parallel_time = data[op][method][largest_idx]
            if parallel_time > 0:
                speedup = seq / parallel_time
            else:
                speedup = 1.0
            row.append(speedup)
        speedup_data.append(row)
    
    speedup_data = np.array(speedup_data)
    
    im = ax.imshow(speedup_data, cmap='RdYlGn', vmin=0, vmax=10)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(['std::\nthread', 'OMP\n1-loop', 'OMP\n2-loop', 'OMP\n3-loop', 'OMP\nTile', 'OMP\nCubed'], fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_title('Speedup at Largest Size\n(Higher = Better)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Method', fontsize=11)
    
    # Add value labels
    for i in range(len(y_labels)):
        for j in range(len(methods)):
            text = ax.text(j, i, f'{speedup_data[i, j]:.2f}x',
                          ha='center', va='center', color='black', fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Speedup (Sequential / Parallel)')
    
    # Right: Relative efficiency
    ax2 = axes[1]
    
    # Calculate average speedup across all sizes
    avg_speedup = []
    for op in operations:
        if op not in data:
            continue
        row = []
        seq = np.array(data[op]['sequential'])
        seq[seq == 0] = 1e-10
        for method in methods:
            parallel = np.array(data[op][method])
            parallel[parallel == 0] = 1e-10
            avg = np.mean(seq / parallel)
            row.append(avg)
        avg_speedup.append(row)
    
    avg_speedup = np.array(avg_speedup)
    
    im2 = ax2.imshow(avg_speedup, cmap='RdYlGn', vmin=0, vmax=10)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(['std::\nthread', 'OMP\n1-loop', 'OMP\n2-loop', 'OMP\n3-loop', 'OMP\nTile', 'OMP\nCubed'], fontsize=8)
    ax2.set_yticks(range(len(y_labels)))
    ax2.set_yticklabels(y_labels, fontsize=9)
    ax2.set_title('Average Speedup (All Sizes)\n(Higher = Better)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Method', fontsize=11)
    
    for i in range(len(y_labels)):
        for j in range(len(methods)):
            text = ax2.text(j, i, f'{avg_speedup[i, j]:.2f}x',
                           ha='center', va='center', color='black', fontsize=8)
    
    plt.colorbar(im2, ax=ax2, label='Average Speedup')
    
    plt.suptitle('Speedup Analysis: Parallel Methods vs Sequential\n' + 
                 '(Values show how many times faster than sequential)',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_comprehensive_dashboard(data: dict, output: str | None = None) -> None:
    """Create a comprehensive dashboard."""
    operations = list(data.keys())
    n_ops = len(operations)
    
    fig = plt.figure(figsize=(24, 16))
    
    # Top row: Individual operation plots
    for idx, op in enumerate(operations):
        ax = fig.add_subplot(2, (n_ops+1)//2, idx+1)
        
        sizes = data[op]['sizes']
        x = np.arange(len(sizes))
        width = 0.11
        
        methods = [
            ('sequential', data[op]['sequential']),
            ('std_thread', data[op]['std_thread']),
            ('omp_1loop', data[op]['omp_1loop']),
            ('omp_2loop', data[op]['omp_2loop']),
            ('omp_3loop', data[op]['omp_3loop']),
            ('omp_tile', data[op]['omp_tile']),
            ('omp_cubed', data[op]['omp_cubed'])
        ]
        
        for i, (method, times) in enumerate(methods):
            if all(t == 0 for t in times):
                continue
            offset = (i - len(methods)/2 + 0.5) * width
            ax.bar(x + offset, times, width, label=LABELS.get(method, method),
                   color=COLORS.get(method, '#999999'), alpha=0.85)
        
        ax.set_xlabel('Size (LAT×LON)', fontsize=11)
        ax.set_ylabel('Time (µs)', fontsize=11)
        ax.set_title(op.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}×{s}' for s in sizes], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_yscale('log')
        ax.legend(loc='upper left', ncol=4, fontsize=8)
    
    # Bottom row: Speedup heatmap and summary
    # Speedup heatmap
    ax = fig.add_subplot(3, 3, 7)
    methods_short = ['std_thread', 'omp_1loop', 'omp_2loop', 'omp_3loop', 'omp_tile', 'omp_cubed']
    largest_idx = -1
    speedup_data = []
    
    for op in operations:
        if op not in data:
            continue
        seq = data[op]['sequential'][largest_idx]
        row = []
        for method in methods_short:
            parallel_time = data[op][method][largest_idx]
            if parallel_time > 0:
                speedup = seq / parallel_time
            else:
                speedup = 1.0
            row.append(speedup)
        speedup_data.append(row)
    
    speedup_data = np.array(speedup_data)
    
    im = ax.imshow(speedup_data, cmap='RdYlGn', vmin=0, vmax=10)
    ax.set_xticks(range(len(methods_short)))
    ax.set_xticklabels(['std::th', '1-loop', '2-loop', '3-loop', 'Tile', 'Cubed'], rotation=30, ha='right', fontsize=8)
    ax.set_yticks(range(len(operations)))
    ax.set_yticklabels([op.replace('_', ' ') for op in operations], fontsize=9)
    ax.set_title('Speedup at Max Size', fontsize=11, fontweight='bold')
    
    for i in range(len(operations)):
        for j in range(len(methods_short)):
            ax.text(j, i, f'{speedup_data[i, j]:.2f}x', ha='center', va='center', fontsize=7)
    
    plt.colorbar(im, ax=ax, label='Speedup')
    
    # Best method per operation
    ax = fig.add_subplot(3, 3, 8)
    ax.axis('off')
    
    table_data = []
    for op in operations:
        if op not in data:
            continue
        
        # Find best method at largest size
        best_method = 'sequential'
        best_time = data[op]['sequential'][-1]
        
        for method in methods_short:
            t = data[op][method][-1]
            if t > 0 and t < best_time:
                best_time = t
                best_method = method
        
        seq_time = data[op]['sequential'][-1]
        speedup = seq_time / best_time if best_time > 0 else 1.0
        
        table_data.append([
            op.replace('_', ' ').title(),
            f'{seq_time:.1f}',
            LABELS.get(best_method, best_method),
            f'{best_time:.1f}',
            f'{speedup:.2f}x'
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Operation', 'Seq Time\n(µs)', 'Best Method', 'Best Time\n(µs)', 'Speedup'],
        loc='center',
        cellLoc='center',
        colColours=['#34495e', '#e74c3c', '#2ecc71', '#27ae60', '#1abc9c']
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    
    ax.set_title('Best Method Summary', fontsize=12, fontweight='bold', pad=30)
    
    # Overall statistics
    ax = fig.add_subplot(3, 3, 9)
    ax.axis('off')
    
    stats_text = "Benchmark Statistics\n\n"
    for op in operations:
        if op not in data:
            continue
        sizes = data[op]['sizes']
        seq_max = data[op]['sequential'][-1]
        best_min = min(data[op][m][-1] for m in methods_short if data[op][m][-1] > 0)
        best_speedup = seq_max / best_min if best_min > 0 else 1.0
        stats_text += f"{op.replace('_', ' ').title()}:\n"
        stats_text += f"  Max size: {sizes[-1]}×{sizes[-1]}\n"
        stats_text += f"  Best speedup: {best_speedup:.2f}x\n\n"
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('OLAP Benchmark Dashboard: Parallelization Strategy Analysis\n' + 
                 '(All times in microseconds, lower is better)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()


def print_summary(data: dict) -> None:
    """Print text summary."""
    print("\n" + "=" * 120)
    print("OLAP BENCHMARK SUMMARY: Parallelization Strategy Comparison")
    print("=" * 120)
    
    for op, op_data in data.items():
        print(f"\n{op.upper().replace('_', ' ')}")
        print("-" * 120)
        print(f"{'Size':<12} {'Sequential':<12} {'std::th':<10} {'1-loop':<10} {'2-loop':<10} {'3-loop':<10} {'Tile':<10} {'Cubed':<10}")
        print("-" * 120)
        
        for i, size in enumerate(op_data['sizes']):
            print(f"{size}×{size:<8} {op_data['sequential'][i]:<12.2f} "
                  f"{op_data['std_thread'][i]:<10.2f} "
                  f"{op_data['omp_1loop'][i]:<10.2f} "
                  f"{op_data['omp_2loop'][i]:<10.2f} "
                  f"{op_data['omp_3loop'][i]:<10.2f} "
                  f"{op_data['omp_tile'][i]:<10.2f} "
                  f"{op_data['omp_cubed'][i]:<10.2f}")
    
    print("\n" + "=" * 120)
    print("All times in MICROSECONDS (µs)")
    print("=" * 120 + "\n")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Loading benchmark data...")
    data = load_all_benchmarks()
    
    if not data:
        print("\nError: No benchmark files found!")
        print("Run the benchmark first: ./build/gpmcube (select option 4)")
        sys.exit(1)
    
    # Print summary
    print_summary(data)
    
    # Generate plots
    print("Generating visualizations...")
    
    # Individual plots for each operation
    for op, op_data in data.items():
        plot_operation_comparison(op_data, op, f"viz_{op}_benchmark.png")
    
    # Grid of all operations
    plot_all_operations_grid(data, "viz_all_operations_grid.png")
    
    # Method comparison across operations
    plot_method_comparison_across_ops(data, "viz_method_comparison.png")
    
    # Speedup heatmap
    plot_speedup_heatmap(data, "viz_speedup_heatmap.png")
    
    # Comprehensive dashboard
    plot_comprehensive_dashboard(data, "viz_comprehensive_dashboard.png")
    
    print("\n" + "=" * 120)
    print("Visualization complete!")
    print("Generated:")
    for op in data.keys():
        print(f"  - viz_{op}_benchmark.png")
    print("  - viz_all_operations_grid.png")
    print("  - viz_method_comparison.png")
    print("  - viz_speedup_heatmap.png")
    print("  - viz_comprehensive_dashboard.png")
    print("\nAll times are in MICROSECONDS (µs)")
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
