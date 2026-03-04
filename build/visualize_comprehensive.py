#!/usr/bin/env python3
"""
Comprehensive Benchmark Visualization Script
Generates multi-plot visualizations for:
1. All operations comparison (Sequential vs std::thread vs OMP)
2. Chunk size sweep analysis
3. Tile size sweep analysis
4. Speedup comparisons across all methods

All times are displayed in MICROSECONDS (µs)
"""

import csv
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, FuncFormatter


# ============================================================================
# STYLE CONFIGURATION
# ============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Color palette
COLORS = {
    'sequential': '#e74c3c',      # Red
    'std_thread': '#3498db',       # Blue
    'omp_default': '#9b59b6',      # Purple
    'omp_chunk': '#2ecc71',        # Green
    'omp_tile': '#f39c12',         # Orange
    'grid': '#ecf0f1'
}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def load_all_operations_csv(filename: str) -> dict:
    """Load benchmark_all_operations.csv (times in microseconds)"""
    data = {
        'operation': [],
        'sequential': [],
        'std_thread': [],
        'omp_default': [],
        'omp_chunk': [],
        'omp_tile': []
    }
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['operation'].append(row['operation'])
            data['sequential'].append(float(row['sequential']))
            data['std_thread'].append(float(row['std_thread']))
            data['omp_default'].append(float(row['omp_default']))
            data['omp_chunk'].append(float(row['omp_chunk']))
            data['omp_tile'].append(float(row['omp_tile']))
    
    return data


def load_chunk_size_sweep_csv(filename: str) -> dict:
    """Load benchmark_chunk_size_sweep.csv (times in microseconds)"""
    data = {
        'chunk_size': [],
        'slice_omp_chunk': [],
        'dice_omp_chunk': [],
        'rollup_mean_omp_chunk': [],
        'rollup_sum_omp_chunk': [],
        'global_mean_omp_chunk': [],
        # Baselines
        'slice_seq': [],
        'slice_std': [],
        'slice_omp_default': [],
        'slice_omp_tile': [],
        'dice_seq': [],
        'dice_std': [],
        'dice_omp_default': [],
        'rollup_mean_seq': [],
        'rollup_mean_std': [],
        'rollup_mean_omp_default': [],
        'rollup_sum_seq': [],
        'rollup_sum_std': [],
        'rollup_sum_omp_default': [],
        'global_mean_seq': [],
        'global_mean_std': [],
        'global_mean_omp_default': []
    }
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['chunk_size'].append(int(row['chunk_size']))
            data['slice_omp_chunk'].append(float(row['slice_omp_chunk']))
            data['dice_omp_chunk'].append(float(row['dice_omp_chunk']))
            data['rollup_mean_omp_chunk'].append(float(row['rollup_mean_omp_chunk']))
            data['rollup_sum_omp_chunk'].append(float(row['rollup_sum_omp_chunk']))
            data['global_mean_omp_chunk'].append(float(row['global_mean_omp_chunk']))
            # Baselines (same for all rows, take first)
            if len(data['chunk_size']) == 1:
                data['slice_seq'].append(float(row['slice_seq']))
                data['slice_std'].append(float(row['slice_std']))
                data['slice_omp_default'].append(float(row['slice_omp_default']))
                data['slice_omp_tile'].append(float(row['slice_omp_tile']))
    
    # Fill baselines for all rows
    for i in range(1, len(data['chunk_size'])):
        for key in ['slice_seq', 'slice_std', 'slice_omp_default', 'slice_omp_tile',
                    'dice_seq', 'dice_std', 'dice_omp_default',
                    'rollup_mean_seq', 'rollup_mean_std', 'rollup_mean_omp_default',
                    'rollup_sum_seq', 'rollup_sum_std', 'rollup_sum_omp_default',
                    'global_mean_seq', 'global_mean_std', 'global_mean_omp_default']:
            if key in data and len(data[key]) > 0:
                data[key].append(data[key][0])
    
    return data


def load_tile_size_sweep_csv(filename: str) -> dict:
    """Load benchmark_tile_size_sweep.csv (times in microseconds)"""
    data = {
        'tile_size': [],
        'slice_omp_tile': [],
        'slice_seq': [],
        'slice_std': [],
        'slice_omp_default': [],
        'slice_omp_chunk': []
    }
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data['tile_size'].append(int(row['tile_size']))
            data['slice_omp_tile'].append(float(row['slice_omp_tile']))
            data['slice_seq'].append(float(row['slice_seq']))
            data['slice_std'].append(float(row['slice_std']))
            data['slice_omp_default'].append(float(row['slice_omp_default']))
            data['slice_omp_chunk'].append(float(row['slice_omp_chunk']))
    
    return data


def load_benchmark_metadata(filename: str) -> dict:
    """Load benchmark_metadata.csv to get constant parameters"""
    metadata = {}
    
    if not os.path.exists(filename):
        # Return defaults if metadata file doesn't exist
        return {
            'cube_dimensions': 'Unknown',
            'chunk_sizes_tested': '1,4,8,16,32,64,128,256',
            'tile_sizes_tested': '8,16,24,32,48,64,96,128',
            'tile_size_fixed': '32',
            'chunk_size_fixed': '64',
            'trials_per_measurement': '5'
        }
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line or line.startswith('key,'):
                continue
            if ',' in line:
                key, value = line.split(',', 1)
                # Remove quotes if present
                value = value.strip('"')
                metadata[key] = value
    
    return metadata


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_all_operations_comparison(data: dict, metadata: dict, output: str | None = None) -> None:
    """Plot comparison of all implementations across all operations."""
    operations = data['operation']
    x = np.arange(len(operations))
    width = 0.18
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top: Execution Time Comparison
    ax1 = axes[0]
    
    bars1 = ax1.bar(x - 2*width, data['sequential'], width, label='Sequential',
                    color=COLORS['sequential'], alpha=0.85, edgecolor='black', linewidth=0.8)
    bars2 = ax1.bar(x - width, data['std_thread'], width, label='std::thread',
                    color=COLORS['std_thread'], alpha=0.85, edgecolor='black', linewidth=0.8)
    bars3 = ax1.bar(x, data['omp_default'], width, label='OMP (default)',
                    color=COLORS['omp_default'], alpha=0.85, edgecolor='black', linewidth=0.8)
    bars4 = ax1.bar(x + width, data['omp_chunk'], width, label='OMP (chunk=64)',
                    color=COLORS['omp_chunk'], alpha=0.85, edgecolor='black', linewidth=0.8)
    bars5 = ax1.bar(x + 2*width, data['omp_tile'], width, label='OMP (tile=32×32)',
                    color=COLORS['omp_tile'], alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax1.set_ylabel('Time (µs)', fontsize=11)
    ax1.set_title('Execution Time: All Operations & Implementations\n' + 
                  f'(Constant: tile_size=32 for chunk tests, chunk_size=64 for tile tests)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(operations, rotation=20, ha='right')
    ax1.legend(loc='upper right', ncol=3, fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    
    # Bottom: Speedup Comparison (relative to sequential)
    ax2 = axes[1]
    
    seq = np.array(data['sequential'])
    seq[seq == 0] = 1e-10  # Avoid division by zero
    
    speedup_std = np.array(data['sequential']) / np.array(data['std_thread'])
    speedup_omp_def = np.array(data['sequential']) / np.array(data['omp_default'])
    speedup_omp_chunk = np.array(data['sequential']) / np.array(data['omp_chunk'])
    speedup_omp_tile = np.array(data['sequential']) / np.array(data['omp_tile'])
    
    bars_s1 = ax2.bar(x - width/2, speedup_std, width/2, label='std::thread',
                      color=COLORS['std_thread'], alpha=0.85, edgecolor='black', linewidth=0.8)
    bars_s2 = ax2.bar(x, speedup_omp_def, width/2, label='OMP (default)',
                      color=COLORS['omp_default'], alpha=0.85, edgecolor='black', linewidth=0.8)
    bars_s3 = ax2.bar(x + width/2, speedup_omp_chunk, width/2, label='OMP (chunk)',
                      color=COLORS['omp_chunk'], alpha=0.85, edgecolor='black', linewidth=0.8)
    bars_s4 = ax2.bar(x + width, speedup_omp_tile, width/2, label='OMP (tile)',
                      color=COLORS['omp_tile'], alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Break-even (1x)')
    ax2.set_ylabel('Speedup (Sequential / Parallel)', fontsize=11)
    ax2.set_xlabel('Operation', fontsize=11)
    ax2.set_title('Speedup Comparison: Parallel vs Sequential', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(operations, rotation=20, ha='right')
    ax2.legend(loc='upper left', ncol=3, fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add metadata text box
    metadata_text = f"Cube: {metadata.get('cube_dimensions', 'N/A')}\n"
    metadata_text += f"Trials: {metadata.get('trials_per_measurement', '5')}"
    fig.text(0.01, 0.99, metadata_text, fontsize=8, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('OLAP Operations: Sequential vs std::thread vs OpenMP\n' + 
                 f'(Times in microseconds, lower is better)', 
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_chunk_size_sweep(data: dict, metadata: dict, output: str | None = None) -> None:
    """Plot chunk size sweep analysis."""
    chunk_sizes = data['chunk_size']
    x = np.arange(len(chunk_sizes))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    operations = [
        ('slice_omp_chunk', 'Slice'),
        ('dice_omp_chunk', 'Dice'),
        ('rollup_mean_omp_chunk', 'Rollup Mean'),
        ('rollup_sum_omp_chunk', 'Rollup Sum'),
        ('global_mean_omp_chunk', 'Global Mean')
    ]
    
    tile_size_fixed = metadata.get('tile_size_fixed', '32')
    
    for idx, (key, title) in enumerate(operations):
        ax = axes[idx]
        
        # Plot OMP with chunk size
        ax.plot(chunk_sizes, data[key], 'o-', linewidth=2, markersize=8, 
                color=COLORS['omp_chunk'], label='OMP (chunk)', markerfacecolor='white', markeredgewidth=1.5)
        
        # Plot baselines (only on first plot to avoid clutter)
        if idx == 0:
            ax.axhline(y=data['slice_seq'][0], color=COLORS['sequential'], linestyle='--', 
                       linewidth=2, label='Sequential', alpha=0.7)
            ax.axhline(y=data['slice_std'][0], color=COLORS['std_thread'], linestyle='--', 
                       linewidth=2, label='std::thread', alpha=0.7)
            ax.axhline(y=data['slice_omp_default'][0], color=COLORS['omp_default'], linestyle=':', 
                       linewidth=2, label='OMP (default)', alpha=0.7)
            ax.legend(loc='upper right', fontsize=9)
        
        ax.set_xlabel('Chunk Size', fontsize=10)
        ax.set_ylabel('Time (µs)', fontsize=10)
        ax.set_title(f'{title}: Chunk Size Impact\n(Fixed: tile_size={tile_size_fixed}×{tile_size_fixed})', 
                     fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xscale('log', base=2)
        
        # Format x-axis to show actual chunk sizes
        ax.set_xticks(chunk_sizes)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
    
    # Last subplot: Summary - best chunk size for each operation
    ax = axes[5]
    best_chunks = []
    best_times = []
    for key, title in operations:
        times = data[key]
        best_idx = np.argmin(times)
        best_chunks.append(chunk_sizes[best_idx])
        best_times.append(times[best_idx])
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(operations)))
    ax.bar(range(len(operations)), best_times, color=colors, alpha=0.8, edgecolor='black')
    for i, (bc, bt) in enumerate(zip(best_chunks, best_times)):
        ax.annotate(f'CS={bc}\n{bt:.1f}µs', (i, bt), ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(range(len(operations)))
    ax.set_xticklabels([op[1] for op in operations], rotation=25, ha='right')
    ax.set_ylabel('Best Time (µs)', fontsize=10)
    ax.set_title('Optimal Chunk Size per Operation', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add metadata
    metadata_text = f"Cube: {metadata.get('cube_dimensions', 'N/A')}\n"
    metadata_text += f"Tile size (fixed): {tile_size_fixed}×{tile_size_fixed}\n"
    metadata_text += f"Chunk sizes tested: {', '.join(map(str, chunk_sizes))}"
    fig.text(0.01, 0.99, metadata_text, fontsize=8, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Chunk Size Sweep Analysis: Performance vs Chunk Size\n' + 
                 '(Times in microseconds, lower is better)', 
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_tile_size_sweep(data: dict, metadata: dict, output: str | None = None) -> None:
    """Plot tile size sweep analysis."""
    tile_sizes = data['tile_size']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Execution Time vs Tile Size
    ax1 = axes[0]
    
    ax1.plot(tile_sizes, data['slice_omp_tile'], 's-', linewidth=2, markersize=10,
             color=COLORS['omp_tile'], label='OMP (tile)', markerfacecolor='white', markeredgewidth=1.5)
    
    # Baselines
    ax1.axhline(y=data['slice_seq'][0], color=COLORS['sequential'], linestyle='--', 
                linewidth=2, label='Sequential', alpha=0.7)
    ax1.axhline(y=data['slice_std'][0], color=COLORS['std_thread'], linestyle='--', 
                linewidth=2, label='std::thread', alpha=0.7)
    ax1.axhline(y=data['slice_omp_default'][0], color=COLORS['omp_default'], linestyle=':', 
                linewidth=2, label='OMP (default)', alpha=0.7)
    ax1.axhline(y=data['slice_omp_chunk'][0], color=COLORS['omp_chunk'], linestyle='-.', 
                linewidth=2, label='OMP (chunk)', alpha=0.7)
    
    ax1.set_xlabel('Tile Size (width × width)', fontsize=11)
    ax1.set_ylabel('Time (µs)', fontsize=11)
    ax1.set_title('Slice Operation: Tile Size Impact\n(Fixed: chunk_size=64)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.set_xticks(tile_sizes)
    
    # Right: Speedup vs Tile Size
    ax2 = axes[1]
    
    seq = data['slice_seq'][0]
    speedup_tile = [seq / t for t in data['slice_omp_tile']]
    
    ax2.plot(tile_sizes, speedup_tile, 's-', linewidth=2, markersize=10,
             color=COLORS['omp_tile'], label='OMP (tile)', markerfacecolor='white', markeredgewidth=1.5)
    
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Break-even (1x)', alpha=0.7)
    
    ax2.set_xlabel('Tile Size (width × width)', fontsize=11)
    ax2.set_ylabel('Speedup (Sequential / OMP Tile)', fontsize=11)
    ax2.set_title('Tile Size: Speedup Analysis', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_xticks(tile_sizes)
    
    # Annotate best tile size
    best_idx = np.argmax(speedup_tile)
    best_tile = tile_sizes[best_idx]
    best_speedup = speedup_tile[best_idx]
    ax2.annotate(f'Best: {best_tile}×{best_tile}\n{best_speedup:.2f}x speedup',
                 xy=(best_tile, best_speedup), xytext=(best_tile*1.15, best_speedup*0.85),
                 fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', color='black'))
    
    # Add metadata
    chunk_size_fixed = metadata.get('chunk_size_fixed', '64')
    metadata_text = f"Cube: {metadata.get('cube_dimensions', 'N/A')}\n"
    metadata_text += f"Chunk size (fixed): {chunk_size_fixed}\n"
    metadata_text += f"Tile sizes tested: {', '.join(map(str, tile_sizes))}"
    fig.text(0.01, 0.99, metadata_text, fontsize=8, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Tile Size Sweep Analysis: Slice Operation\n' + 
                 '(Times in microseconds, lower is better)', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()


def plot_comprehensive_dashboard(data_ops: dict, data_chunk: dict, data_tile: dict, 
                                  metadata: dict, output: str | None = None) -> None:
    """Create a comprehensive dashboard with all visualizations."""
    fig = plt.figure(figsize=(20, 14))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Operations comparison (top left - 2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    operations = data_ops['operation']
    x = np.arange(len(operations))
    width = 0.18
    
    ax1.bar(x - 2*width, data_ops['sequential'], width, label='Sequential',
            color=COLORS['sequential'], alpha=0.85)
    ax1.bar(x - width, data_ops['std_thread'], width, label='std::thread',
            color=COLORS['std_thread'], alpha=0.85)
    ax1.bar(x, data_ops['omp_default'], width, label='OMP (default)',
            color=COLORS['omp_default'], alpha=0.85)
    ax1.bar(x + width, data_ops['omp_chunk'], width, label='OMP (chunk)',
            color=COLORS['omp_chunk'], alpha=0.85)
    ax1.bar(x + 2*width, data_ops['omp_tile'], width, label='OMP (tile)',
            color=COLORS['omp_tile'], alpha=0.85)
    
    ax1.set_ylabel('Time (µs)', fontsize=11)
    ax1.set_title('All Operations: Implementation Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(operations, rotation=20, ha='right')
    ax1.legend(loc='upper right', ncol=3, fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    
    # 2. Speedup comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    seq = np.array(data_ops['sequential'])
    seq[seq == 0] = 1e-10
    speedup_std = np.array(data_ops['sequential']) / np.array(data_ops['std_thread'])
    speedup_omp_def = np.array(data_ops['sequential']) / np.array(data_ops['omp_default'])
    speedup_omp_chunk = np.array(data_ops['sequential']) / np.array(data_ops['omp_chunk'])
    speedup_omp_tile = np.array(data_ops['sequential']) / np.array(data_ops['omp_tile'])
    
    x_sp = np.arange(len(operations))
    width_sp = 0.2
    ax2.bar(x_sp - 1.5*width_sp, speedup_std, width_sp, label='std::thread', color=COLORS['std_thread'], alpha=0.85)
    ax2.bar(x_sp - 0.5*width_sp, speedup_omp_def, width_sp, label='OMP (default)', color=COLORS['omp_default'], alpha=0.85)
    ax2.bar(x_sp + 0.5*width_sp, speedup_omp_chunk, width_sp, label='OMP (chunk)', color=COLORS['omp_chunk'], alpha=0.85)
    ax2.bar(x_sp + 1.5*width_sp, speedup_omp_tile, width_sp, label='OMP (tile)', color=COLORS['omp_tile'], alpha=0.85)
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Break-even')
    ax2.set_ylabel('Speedup', fontsize=10)
    ax2.set_title('Speedup Overview', fontsize=11, fontweight='bold')
    ax2.set_xticks(x_sp)
    ax2.set_xticklabels(operations, rotation=25, ha='right')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 3. Chunk size sweep (middle left)
    ax3 = fig.add_subplot(gs[1, 2])
    chunk_sizes = data_chunk['chunk_size']
    ax3.plot(chunk_sizes, data_chunk['slice_omp_chunk'], 'o-', linewidth=2, 
             color=COLORS['omp_chunk'], label='Slice', markerfacecolor='white')
    ax3.plot(chunk_sizes, data_chunk['dice_omp_chunk'], 's-', linewidth=2, 
             color=COLORS['std_thread'], label='Dice', markerfacecolor='white')
    ax3.plot(chunk_sizes, data_chunk['rollup_mean_omp_chunk'], '^-', linewidth=2, 
             color=COLORS['omp_default'], label='Rollup Mean', markerfacecolor='white')
    ax3.set_xlabel('Chunk Size', fontsize=10)
    ax3.set_ylabel('Time (µs)', fontsize=10)
    ax3.set_title('Chunk Size Impact', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(chunk_sizes)
    ax3.get_xaxis().set_major_formatter(ScalarFormatter())
    
    # 4. Tile size sweep (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    tile_sizes = data_tile['tile_size']
    ax4.plot(tile_sizes, data_tile['slice_omp_tile'], 's-', linewidth=2, 
             color=COLORS['omp_tile'], label='OMP (tile)', markerfacecolor='white', markeredgewidth=1.5)
    ax4.axhline(y=data_tile['slice_seq'][0], color=COLORS['sequential'], linestyle='--', 
                linewidth=2, label='Sequential', alpha=0.7)
    ax4.axhline(y=data_tile['slice_std'][0], color=COLORS['std_thread'], linestyle='--', 
                linewidth=2, label='std::thread', alpha=0.7)
    ax4.axhline(y=data_tile['slice_omp_default'][0], color=COLORS['omp_default'], linestyle=':', 
                linewidth=2, label='OMP (default)', alpha=0.7)
    ax4.set_xlabel('Tile Size', fontsize=11)
    ax4.set_ylabel('Time (µs)', fontsize=11)
    ax4.set_title('Tile Size Impact (Slice)', fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.set_xticks(tile_sizes)
    
    # 5. Best parameters summary (bottom middle)
    ax5 = fig.add_subplot(gs[2, 1])
    ops_summary = ['Slice', 'Dice', 'Rollup\nMean', 'Rollup\nSum', 'Global\nMean']
    keys = ['slice_omp_chunk', 'dice_omp_chunk', 'rollup_mean_omp_chunk', 
            'rollup_sum_omp_chunk', 'global_mean_omp_chunk']
    best_chunks = []
    best_times = []
    for key in keys:
        times = data_chunk[key]
        best_idx = np.argmin(times)
        best_chunks.append(data_chunk['chunk_size'][best_idx])
        best_times.append(times[best_idx])
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ops_summary)))
    bars = ax5.bar(range(len(ops_summary)), best_times, color=colors, alpha=0.8, edgecolor='black')
    for i, (bc, bt) in enumerate(zip(best_chunks, best_times)):
        ax5.annotate(f'CS={bc}', (i, bt), ha='center', va='bottom', fontsize=8)
    
    ax5.set_xticks(range(len(ops_summary)))
    ax5.set_xticklabels(ops_summary)
    ax5.set_ylabel('Best Time (µs)', fontsize=10)
    ax5.set_title('Optimal Chunk Size', fontsize=11, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 6. Tile size speedup (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    seq = data_tile['slice_seq'][0]
    speedup_tile = [seq / t for t in data_tile['slice_omp_tile']]
    ax6.plot(tile_sizes, speedup_tile, 's-', linewidth=2, color=COLORS['omp_tile'], 
             markerfacecolor='white', markeredgewidth=1.5)
    ax6.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Break-even')
    best_idx = np.argmax(speedup_tile)
    ax6.annotate(f'Best: {tile_sizes[best_idx]}×{tile_sizes[best_idx]}\n{speedup_tile[best_idx]:.2f}x',
                 xy=(tile_sizes[best_idx], speedup_tile[best_idx]),
                 xytext=(tile_sizes[best_idx]*1.15, speedup_tile[best_idx]*0.85),
                 fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', color='black'))
    ax6.set_xlabel('Tile Size', fontsize=10)
    ax6.set_ylabel('Speedup', fontsize=10)
    ax6.set_title('Tile Size Speedup', fontsize=11, fontweight='bold')
    ax6.grid(alpha=0.3, linestyle='--')
    ax6.set_xticks(tile_sizes)
    
    # Add comprehensive metadata
    metadata_text = f"Cube: {metadata.get('cube_dimensions', 'N/A')}\n"
    metadata_text += f"Trials: {metadata.get('trials_per_measurement', '5')}\n"
    metadata_text += f"Chunk sizes: {metadata.get('chunk_sizes_tested', 'N/A')}\n"
    metadata_text += f"Tile sizes: {metadata.get('tile_sizes_tested', 'N/A')}"
    fig.text(0.01, 0.99, metadata_text, fontsize=8, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Comprehensive OLAP Benchmark Dashboard: Sequential vs std::thread vs OpenMP\n' + 
                 '(All times in microseconds)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')
        print(f"Saved: {output}")
    else:
        plt.show()


def print_summary(data_ops: dict, data_chunk: dict, data_tile: dict, metadata: dict) -> None:
    """Print text summary of benchmark results."""
    print("\n" + "=" * 90)
    print("COMPREHENSIVE BENCHMARK SUMMARY: Sequential vs std::thread vs OpenMP")
    print("=" * 90)
    
    print(f"\nCube Dimensions: {metadata.get('cube_dimensions', 'N/A')}")
    print(f"Trials per Measurement: {metadata.get('trials_per_measurement', '5')}")
    print(f"Time Unit: microseconds (µs)")
    
    print("\n[1] OPERATIONS COMPARISON")
    print("-" * 90)
    print(f"{'Operation':<15} {'Sequential':<12} {'std::thread':<12} {'OMP(def)':<12} {'OMP(chunk)':<12} {'OMP(tile)':<12}")
    print("-" * 90)
    
    for i, op in enumerate(data_ops['operation']):
        print(f"{op:<15} {data_ops['sequential'][i]:<12.2f} {data_ops['std_thread'][i]:<12.2f} "
              f"{data_ops['omp_default'][i]:<12.2f} {data_ops['omp_chunk'][i]:<12.2f} "
              f"{data_ops['omp_tile'][i]:<12.2f}")
    
    # Calculate average speedups
    seq = np.array(data_ops['sequential'])
    seq[seq == 0] = 1e-10
    avg_speedup_std = np.mean(seq / np.array(data_ops['std_thread']))
    avg_speedup_omp_def = np.mean(seq / np.array(data_ops['omp_default']))
    avg_speedup_omp_chunk = np.mean(seq / np.array(data_ops['omp_chunk']))
    avg_speedup_omp_tile = np.mean(seq / np.array(data_ops['omp_tile']))
    
    print("-" * 90)
    print(f"{'Avg Speedup:':<15} {'':<12} {avg_speedup_std:<12.2f}x {avg_speedup_omp_def:<12.2f}x "
          f"{avg_speedup_omp_chunk:<12.2f}x {avg_speedup_omp_tile:<12.2f}x")
    
    print("\n[2] CHUNK SIZE ANALYSIS")
    print("-" * 90)
    ops_chunk = [('slice_omp_chunk', 'Slice'), ('dice_omp_chunk', 'Dice'),
                 ('rollup_mean_omp_chunk', 'Rollup Mean'), ('rollup_sum_omp_chunk', 'Rollup Sum'),
                 ('global_mean_omp_chunk', 'Global Mean')]
    
    print(f"{'Operation':<15} {'Best Chunk':<12} {'Best Time':<12} {'Worst Chunk':<12} {'Worst Time':<12}")
    print("-" * 90)
    
    for key, name in ops_chunk:
        times = data_chunk[key]
        best_idx = np.argmin(times)
        worst_idx = np.argmax(times)
        print(f"{name:<15} {data_chunk['chunk_size'][best_idx]:<12} {times[best_idx]:<12.2f} "
              f"{data_chunk['chunk_size'][worst_idx]:<12} {times[worst_idx]:<12.2f}")
    
    print("\n[3] TILE SIZE ANALYSIS (Slice)")
    print("-" * 90)
    best_tile_idx = np.argmin(data_tile['slice_omp_tile'])
    worst_tile_idx = np.argmax(data_tile['slice_omp_tile'])
    print(f"Best Tile Size:  {data_tile['tile_size'][best_tile_idx]} × {data_tile['tile_size'][best_tile_idx]} "
          f"({data_tile['slice_omp_tile'][best_tile_idx]:.2f} µs)")
    print(f"Worst Tile Size: {data_tile['tile_size'][worst_tile_idx]} × {data_tile['tile_size'][worst_tile_idx]} "
          f"({data_tile['slice_omp_tile'][worst_tile_idx]:.2f} µs)")
    print(f"Sequential:      {data_tile['slice_seq'][0]:.2f} µs")
    print(f"Speedup (best):  {data_tile['slice_seq'][0] / data_tile['slice_omp_tile'][best_tile_idx]:.2f}x")
    
    print("\n[4] CONSTANT PARAMETERS")
    print("-" * 90)
    print(f"Fixed tile size (during chunk sweep): {metadata.get('tile_size_fixed', '32')} × {metadata.get('tile_size_fixed', '32')}")
    print(f"Fixed chunk size (during tile sweep): {metadata.get('chunk_size_fixed', '64')}")
    
    print("=" * 90 + "\n")


def main():
    # Default filenames
    ops_file = "benchmark_all_operations.csv"
    chunk_file = "benchmark_chunk_size_sweep.csv"
    tile_file = "benchmark_tile_size_sweep.csv"
    metadata_file = "benchmark_metadata.csv"
    
    # Output files
    output_ops = "viz_operations_comparison.png"
    output_chunk = "viz_chunk_size_sweep.png"
    output_tile = "viz_tile_size_sweep.png"
    output_dashboard = "viz_comprehensive_dashboard.png"
    
    # Parse arguments
    if len(sys.argv) > 1:
        ops_file = sys.argv[1]
    if len(sys.argv) > 2:
        chunk_file = sys.argv[2]
    if len(sys.argv) > 3:
        tile_file = sys.argv[3]
    if len(sys.argv) > 4:
        metadata_file = sys.argv[4]
    
    # Check if files exist
    missing_files = []
    for f in [ops_file, chunk_file, tile_file]:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        print(f"Error: Missing files: {', '.join(missing_files)}")
        print("Run the comprehensive benchmark first: ./build/gpmcube (select option 4)")
        sys.exit(1)
    
    # Load data
    print("Loading benchmark data...")
    data_ops = load_all_operations_csv(ops_file)
    data_chunk = load_chunk_size_sweep_csv(chunk_file)
    data_tile = load_tile_size_sweep_csv(tile_file)
    metadata = load_benchmark_metadata(metadata_file)
    
    # Print summary
    print_summary(data_ops, data_chunk, data_tile, metadata)
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_all_operations_comparison(data_ops, metadata, output_ops)
    plot_chunk_size_sweep(data_chunk, metadata, output_chunk)
    plot_tile_size_sweep(data_tile, metadata, output_tile)
    plot_comprehensive_dashboard(data_ops, data_chunk, data_tile, metadata, output_dashboard)
    
    print("\n" + "=" * 90)
    print("Visualization complete!")
    print(f"  - {output_ops} (operations comparison)")
    print(f"  - {output_chunk} (chunk size sweep)")
    print(f"  - {output_tile} (tile size sweep)")
    print(f"  - {output_dashboard} (comprehensive dashboard)")
    print("\nAll times are in MICROSECONDS (µs)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
