#!/usr/bin/env python3
"""
Size Sweep Benchmark Visualization
Compares: Sequential vs std::thread vs OMP (row_chunk, col_chunk, tile, cubed)
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
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 15

# Color palette - 6 distinct colors
COLORS = {
    'sequential': '#e74c3c',      # Red
    'std_thread': '#3498db',       # Blue
    'omp_row_chunk': '#9b59b6',    # Purple
    'omp_col_chunk': '#e67e22',    # Orange
    'omp_tile': '#2ecc71',         # Green
    'omp_cubed': '#1abc9c'         # Teal
}

LABELS = {
    'sequential': 'Sequential',
    'std_thread': 'std::thread',
    'omp_row_chunk': 'OMP (row, chunk=64)',
    'omp_col_chunk': 'OMP (col, chunk=64)',
    'omp_tile': 'OMP (tile=32×32)',
    'omp_cubed': 'OMP (cube=16×16×16)'
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
        'omp_row_chunk': [],
        'omp_col_chunk': [],
        'omp_tile': [],
        'omp_cubed': []
    }
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['sizes'].append(int(row['size']))
            data['sequential'].append(float(row['sequential']))
            data['std_thread'].append(float(row['std_thread']))
            data['omp_row_chunk'].append(float(row['omp_row_chunk']))
            data['omp_col_chunk'].append(float(row['omp_col_chunk']))
            data['omp_tile'].append(float(row['omp_tile']))
            data['omp_cubed'].append(float(row['omp_cubed']))
    
    return data


def load_all_benchmarks() -> dict:
    """Load all benchmark files."""
    files = {
        'slice': 'benchmark_slice_size_sweep.csv',
        'dice': 'benchmark_dice_size_sweep.csv',
        'rollup_mean': 'benchmark_rollup_mean_size_sweep.csv',
        'global_mean': 'benchmark_global_mean_size_sweep.csv'
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
    width = 0.12
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    methods = [
        ('sequential', data['sequential']),
        ('std_thread', data['std_thread']),
        ('omp_row_chunk', data['omp_row_chunk']),
        ('omp_col_chunk', data['omp_col_chunk']),
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
    ax.set_title(f'{operation.replace("_", " ").title()}: Implementation Comparison\n' + 
                 f'(1 time slice, {sizes[0]}×{sizes[0]} to {sizes[-1]}×{sizes[-1]})',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}×{s}' for s in sizes], rotation=25, ha='right')
    ax.legend(loc='upper left', ncol=3, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_all_operations_grid(data: dict, output: str | None = None) -> None:
    """Plot all operations in a 2x2 grid."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    operations = ['slice', 'dice', 'rollup_mean', 'global_mean']
    titles = ['Slice Time (t=0)', 'Dice Time (t=0..10)', 
              'Rollup Time Mean (T=24)', 'Global Mean (T=24)']
    
    for idx, (op, title) in enumerate(zip(operations, titles)):
        if op not in data:
            continue
        
        ax = axes[idx]
        sizes = data[op]['sizes']
        x = np.arange(len(sizes))
        width = 0.12
        
        methods = [
            ('sequential', data[op]['sequential']),
            ('std_thread', data[op]['std_thread']),
            ('omp_row_chunk', data[op]['omp_row_chunk']),
            ('omp_col_chunk', data[op]['omp_col_chunk']),
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
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}×{s}' for s in sizes], rotation=20, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_yscale('log')
    
    # Common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=11, 
               bbox_to_anchor=(0.5, 1.02))
    
    plt.suptitle('OLAP Operations: Size Sweep Benchmark\n' + 
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
    operations = ['slice', 'dice', 'rollup_mean', 'global_mean']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    methods = ['sequential', 'std_thread', 'omp_row_chunk', 'omp_col_chunk', 'omp_tile', 'omp_cubed']
    method_titles = ['Sequential', 'std::thread', 'OMP Row', 'OMP Col', 'OMP Tile', 'OMP Cubed']
    
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
    operations = ['slice', 'dice', 'rollup_mean', 'global_mean']
    methods = ['std_thread', 'omp_row_chunk', 'omp_col_chunk', 'omp_tile', 'omp_cubed']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
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
    ax.set_xticklabels(['std::\nthread', 'OMP\nRow', 'OMP\nCol', 'OMP\nTile', 'OMP\nCubed'], fontsize=9)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_title('Speedup at Largest Size (2048×2048)\n(Higher = Better)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Method', fontsize=11)
    
    # Add value labels
    for i in range(len(y_labels)):
        for j in range(len(methods)):
            text = ax.text(j, i, f'{speedup_data[i, j]:.2f}x',
                          ha='center', va='center', color='black', fontsize=9)
    
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
    ax2.set_xticklabels(['std::\nthread', 'OMP\nRow', 'OMP\nCol', 'OMP\nTile', 'OMP\nCubed'], fontsize=9)
    ax2.set_yticks(range(len(y_labels)))
    ax2.set_yticklabels(y_labels, fontsize=10)
    ax2.set_title('Average Speedup (All Sizes)\n(Higher = Better)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Method', fontsize=11)
    
    for i in range(len(y_labels)):
        for j in range(len(methods)):
            text = ax2.text(j, i, f'{avg_speedup[i, j]:.2f}x',
                           ha='center', va='center', color='black', fontsize=9)
    
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
    fig = plt.figure(figsize=(22, 14))
    
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    operations = ['slice', 'dice', 'rollup_mean', 'global_mean']
    titles = ['Slice', 'Dice', 'Rollup Mean', 'Global Mean']
    
    # Top 2 rows: Individual operation plots
    for idx, (op, title) in enumerate(zip(operations, titles)):
        if op not in data:
            continue
        
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col*2:(col+1)*2])
        
        sizes = data[op]['sizes']
        x = np.arange(len(sizes))
        width = 0.12
        
        methods = [
            ('sequential', data[op]['sequential']),
            ('std_thread', data[op]['std_thread']),
            ('omp_row_chunk', data[op]['omp_row_chunk']),
            ('omp_col_chunk', data[op]['omp_col_chunk']),
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
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}×{s}' for s in sizes], rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_yscale('log')
        ax.legend(loc='upper left', ncol=3, fontsize=9)
    
    # Bottom row: Speedup heatmap and summary
    # Speedup heatmap
    ax = fig.add_subplot(gs[2, 0:2])
    methods_short = ['std_thread', 'omp_row_chunk', 'omp_col_chunk', 'omp_tile', 'omp_cubed']
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
    ax.set_xticklabels(['std::thread', 'OMP Row', 'OMP Col', 'OMP Tile', 'OMP Cubed'], rotation=25, ha='right', fontsize=9)
    ax.set_yticks(range(len(operations)))
    ax.set_yticklabels([t.replace(' ', '\n') for t in titles], fontsize=10)
    ax.set_title('Speedup at 2048×2048', fontsize=11, fontweight='bold')
    
    for i in range(len(operations)):
        for j in range(len(methods_short)):
            ax.text(j, i, f'{speedup_data[i, j]:.2f}x', ha='center', va='center', fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Speedup')
    
    # Best method per operation
    ax = fig.add_subplot(gs[2, 2:4])
    ax.axis('off')
    
    table_data = []
    for op, title in zip(operations, titles):
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
            title,
            f'{seq_time:.1f}',
            LABELS.get(best_method, best_method).replace(' (', '\n('),
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
    table.set_fontsize(10)
    table.scale(1.3, 1.8)
    
    ax.set_title('Best Method Summary\n(Largest Size: 2048×2048)', fontsize=12, fontweight='bold', pad=30)
    
    plt.suptitle('OLAP Benchmark Dashboard: Size Sweep Analysis\n' + 
                 '(All times in microseconds, lower is better)',
                 fontsize=16, fontweight='bold', y=0.995)
    
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()


def print_summary(data: dict) -> None:
    """Print text summary."""
    print("\n" + "=" * 100)
    print("SIZE SWEEP BENCHMARK SUMMARY")
    print("=" * 100)
    
    for op, op_data in data.items():
        print(f"\n{op.upper().replace('_', ' ')}")
        print("-" * 100)
        print(f"{'Size':<12} {'Sequential':<12} {'std::thread':<12} {'OMP Row':<12} {'OMP Col':<12} {'OMP Tile':<12} {'OMP Cubed':<12}")
        print("-" * 100)
        
        for i, size in enumerate(op_data['sizes']):
            print(f"{size}×{size:<8} {op_data['sequential'][i]:<12.2f} "
                  f"{op_data['std_thread'][i]:<12.2f} "
                  f"{op_data['omp_row_chunk'][i]:<12.2f} "
                  f"{op_data['omp_col_chunk'][i]:<12.2f} "
                  f"{op_data['omp_tile'][i]:<12.2f} "
                  f"{op_data['omp_cubed'][i]:<12.2f}")
    
    print("\n" + "=" * 100)
    print("All times in MICROSECONDS (µs)")
    print("=" * 100 + "\n")


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
        plot_operation_comparison(op_data, op, f"viz_{op}_size_sweep.png")
    
    # 2x2 grid of all operations
    plot_all_operations_grid(data, "viz_all_operations_grid.png")
    
    # Method comparison across operations
    plot_method_comparison_across_ops(data, "viz_method_comparison.png")
    
    # Speedup heatmap
    plot_speedup_heatmap(data, "viz_speedup_heatmap.png")
    
    # Comprehensive dashboard
    plot_comprehensive_dashboard(data, "viz_comprehensive_dashboard.png")
    
    print("\n" + "=" * 100)
    print("Visualization complete!")
    print("Generated:")
    for op in data.keys():
        print(f"  - viz_{op}_size_sweep.png")
    print("  - viz_all_operations_grid.png (2x2 comparison)")
    print("  - viz_method_comparison.png (6 methods across ops)")
    print("  - viz_speedup_heatmap.png (speedup analysis)")
    print("  - viz_comprehensive_dashboard.png (full dashboard)")
    print("\nAll times are in MICROSECONDS (µs)")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
