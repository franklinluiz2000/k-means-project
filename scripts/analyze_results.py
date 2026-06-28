"""
Analise e visualizacao de resultados dos benchmarks K-means.

Le os CSVs gerados pelos scripts de benchmark e gera graficos
para a apresentacao do projeto.

Uso: python scripts/analyze_results.py
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_DIR = os.path.join(PROJECT_DIR, 'results', 'csv')
FIG_DIR = os.path.join(PROJECT_DIR, 'results', 'figures')


def setup() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_benchmark_comparison() -> None:
    csv_path = os.path.join(CSV_DIR, 'benchmark_results.csv')
    if not os.path.exists(csv_path):
        print("  SKIP: benchmark_results.csv nao encontrado")
        return

    df = pd.read_csv(csv_path)
    df = df[df['time_seconds'] != 'NA']
    df['time_seconds'] = df['time_seconds'].astype(float)

    stats = df.groupby('version')['time_seconds'].agg(['mean', 'std']).reset_index()
    stats = stats.sort_values('mean', ascending=False)

    colors = {
        'sequential': '#3498db',
        'mpi_openmp': '#2ecc71',
        'openmp_gpu': '#e67e22',
        'cuda': '#e74c3c',
    }

    fig, ax = plt.subplots()
    bars = ax.barh(
        stats['version'],
        stats['mean'],
        xerr=stats['std'],
        color=[colors.get(v, '#95a5a6') for v in stats['version']],
        capsize=5,
        edgecolor='white',
        linewidth=0.5,
    )

    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(
            bar.get_width() + row['std'] + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{row['mean']:.2f}s",
            va='center',
            fontweight='bold',
        )

    ax.set_xlabel('Tempo (segundos)')
    ax.set_title('Comparacao de Tempo - K-means (70k amostras, K=10)')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'benchmark_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Salvo: {path}")

    if 'sequential' in stats['version'].values:
        seq_time = stats[stats['version'] == 'sequential']['mean'].values[0]
        print("\n  Speedup:")
        for _, row in stats.iterrows():
            if row['version'] != 'sequential':
                speedup = seq_time / row['mean']
                print(f"    {row['version']}: {speedup:.2f}x")


def plot_strong_scaling() -> None:
    csv_path = os.path.join(CSV_DIR, 'strong_scaling.csv')
    if not os.path.exists(csv_path):
        print("  SKIP: strong_scaling.csv nao encontrado")
        return

    df = pd.read_csv(csv_path)
    df = df[df['time_seconds'] != 'NA']
    df['time_seconds'] = df['time_seconds'].astype(float)

    df['label'] = np.where(
        df['total_procs'] == 1,
        'Seq (1)',
        df['nodes'].astype(str) + 'N x ' + df['threads_per_task'].astype(str) + 'T'
    )

    stats = (
        df.groupby(['nodes', 'tasks', 'threads_per_task', 'total_procs', 'label'])
        ['time_seconds']
        .agg(['mean', 'std'])
        .reset_index()
        .sort_values(['nodes', 'threads_per_task'])
    )

    seq_time = stats[stats['total_procs'] == 1]['mean'].values
    if len(seq_time) == 0:
        print("  SKIP: dados sequenciais (1 proc) nao encontrados para speedup")
        return
    seq_time = seq_time[0]

    stats['speedup'] = seq_time / stats['mean']
    stats['efficiency'] = stats['speedup'] / stats['total_procs'] * 100

    node_colors = {1: '#3498db', 2: '#e67e22', 4: '#e74c3c'}
    node_markers = {1: 'o', 2: 's', 4: 'D'}

    # Grafico de Speedup por configuracao
    fig, ax = plt.subplots(figsize=(12, 7))
    for n in sorted(stats['nodes'].unique()):
        subset = stats[stats['nodes'] == n]
        if n == 1 and 1 in subset['total_procs'].values:
            subset = subset[subset['total_procs'] > 1]
        if subset.empty:
            continue
        ax.plot(
            subset['total_procs'], subset['speedup'],
            marker=node_markers.get(n, 'o'), linestyle='-',
            color=node_colors.get(n, '#95a5a6'),
            linewidth=2, markersize=8,
            label=f'{n} no(s) MPI',
        )
    max_p = int(stats['total_procs'].max())
    ideal_range = np.array([1, 2, 4, 8, 16, 32, 64])
    ideal_range = ideal_range[ideal_range <= max_p]
    ax.plot(ideal_range, ideal_range, '--', color='#95a5a6', linewidth=1, label='Ideal (linear)')
    ax.scatter([1], [1], marker='*', s=200, color='black', zorder=5, label='Sequencial')
    ax.set_xlabel('Total de Processadores (nos x threads)')
    ax.set_ylabel('Speedup')
    ax.set_title('Escalabilidade Forte - Speedup (K-means, 70k amostras)')
    ax.set_xscale('log', base=2)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'strong_scaling_speedup.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Salvo: {path}")

    # Grafico de barras com todas as configuracoes
    fig, ax = plt.subplots(figsize=(14, 7))
    labels = stats['label'].values
    times = stats['mean'].values
    stds = stats['std'].values
    colors = [node_colors.get(n, '#95a5a6') for n in stats['nodes']]

    bars = ax.barh(range(len(labels)), times, xerr=stds, color=colors,
                   capsize=4, edgecolor='white', linewidth=0.5)

    for i, (bar, t, sp) in enumerate(zip(bars, times, stats['speedup'])):
        ax.text(
            bar.get_width() + stds[i] + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.1f}s  ({sp:.1f}x)",
            va='center', fontsize=10,
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Tempo (segundos)')
    ax.set_title('Escalabilidade Forte - Tempo por Configuracao')
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'strong_scaling_time.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Salvo: {path}")

    # Grafico de Eficiencia
    fig, ax = plt.subplots(figsize=(12, 7))
    for n in sorted(stats['nodes'].unique()):
        subset = stats[(stats['nodes'] == n) & (stats['total_procs'] > 1)]
        if subset.empty:
            continue
        ax.plot(
            subset['total_procs'], subset['efficiency'],
            marker=node_markers.get(n, 'o'), linestyle='-',
            color=node_colors.get(n, '#95a5a6'),
            linewidth=2, markersize=8,
            label=f'{n} no(s) MPI',
        )
    ax.axhline(y=100, color='#95a5a6', linestyle='--', linewidth=1, label='Ideal (100%)')
    ax.set_xlabel('Total de Processadores')
    ax.set_ylabel('Eficiencia (%)')
    ax.set_title('Escalabilidade Forte - Eficiencia')
    ax.set_xscale('log', base=2)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'strong_scaling_efficiency.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Salvo: {path}")


def plot_weak_scaling() -> None:
    csv_path = os.path.join(CSV_DIR, 'weak_scaling.csv')
    if not os.path.exists(csv_path):
        print("  SKIP: weak_scaling.csv nao encontrado")
        return

    df = pd.read_csv(csv_path)
    df = df[df['time_seconds'] != 'NA']
    df['time_seconds'] = df['time_seconds'].astype(float)

    stats = df.groupby('tasks')['time_seconds'].agg(['mean', 'std']).reset_index()
    stats = stats.sort_values('tasks')

    base_time = stats['mean'].values[0]
    stats['efficiency'] = base_time / stats['mean'] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    tasks = stats['tasks'].values

    ax1.errorbar(tasks, stats['mean'], yerr=stats['std'], fmt='o-', color='#3498db',
                 linewidth=2, markersize=8, capsize=5)
    ax1.axhline(y=base_time, color='#95a5a6', linestyle='--', linewidth=1, label='Ideal (constante)')
    ax1.set_xlabel('Numero de Processadores')
    ax1.set_ylabel('Tempo (segundos)')
    ax1.set_title('Escalabilidade Fraca - Tempo')
    ax1.legend()

    ax2.plot(tasks, stats['efficiency'], 's-', color='#2ecc71', linewidth=2, markersize=8)
    ax2.axhline(y=100, color='#95a5a6', linestyle='--', linewidth=1, label='Ideal (100%)')
    ax2.set_xlabel('Numero de Processadores')
    ax2.set_ylabel('Eficiencia (%)')
    ax2.set_title('Escalabilidade Fraca - Eficiencia')
    ax2.set_ylim(0, 110)
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'weak_scaling.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Salvo: {path}")


def plot_comprehensive_comparison() -> None:
    csv_gpu = os.path.join(CSV_DIR, 'gpu_comprehensive.csv')
    csv_mpi = os.path.join(CSV_DIR, 'mpi_comprehensive.csv')
    
    dfs = []
    if os.path.exists(csv_gpu):
        dfs.append(pd.read_csv(csv_gpu))
    if os.path.exists(csv_mpi):
        dfs.append(pd.read_csv(csv_mpi))
        
    if not dfs:
        print("  SKIP: Nenhum CSV abrangente encontrado.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['time_seconds'] != 'NA']
    df['time_seconds'] = df['time_seconds'].astype(float)
    df['samples'] = df['samples'].astype(int)

    stats = df.groupby(['version', 'samples'])['time_seconds'].agg(['mean', 'std']).reset_index()
    
    # 1. Gráfico Absoluto (Tempo vs Samples)
    fig, ax = plt.subplots(figsize=(12, 7))
    versions = stats['version'].unique()
    colors = {'sequential': '#3498db', 'cuda': '#e74c3c', 'mpi_openmp_64': '#2ecc71', 'openmp_gpu': '#9b59b6'}
    markers = {'sequential': 'o', 'cuda': 'D', 'mpi_openmp_64': 's', 'openmp_gpu': '^'}
    labels = {'sequential': 'Sequencial (1 CPU)', 'cuda': 'CUDA (1 GPU V100)', 'mpi_openmp_64': 'MPI+OpenMP (64 CPUs)', 'openmp_gpu': 'OpenMP (1 GPU V100)'}
    
    for v in versions:
        sub = stats[stats['version'] == v].sort_values('samples')
        ax.plot(sub['samples'], sub['mean'], marker=markers.get(v, 'o'), color=colors.get(v, 'k'), 
                linewidth=2, markersize=8, label=labels.get(v, v))
                
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Tamanho do Dataset (Num. Amostras)')
    ax.set_ylabel('Tempo (segundos) - Log Scale')
    ax.set_title('Comparação Absoluta - Tempo de Execução vs Tamanho dos Dados')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'comprehensive_time.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Salvo: {path}")

    # 2. Gráfico de Speedup (Em relação ao Sequencial)
    seq_stats = stats[stats['version'] == 'sequential'].set_index('samples')['mean']
    if not seq_stats.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        for v in versions:
            if v == 'sequential': continue
            sub = stats[stats['version'] == v].sort_values('samples')
            speedup = []
            for _, row in sub.iterrows():
                if row['samples'] in seq_stats.index:
                    speedup.append(seq_stats[row['samples']] / row['mean'])
                else:
                    speedup.append(np.nan)
            
            ax.plot(sub['samples'], speedup, marker=markers.get(v, 'o'), color=colors.get(v, 'k'),
                    linewidth=2, markersize=8, label=labels.get(v, v))
                    
        ax.set_xscale('log')
        ax.axhline(y=1, color='#95a5a6', linestyle='--', linewidth=1, label='Baseline Sequencial (1x)')
        ax.set_xlabel('Tamanho do Dataset (Num. Amostras)')
        ax.set_ylabel('Speedup (vezes mais rápido)')
        ax.set_title('Speedup vs Tamanho dos Dados (Baseline: Sequencial)')
        ax.legend()
        plt.tight_layout()
        path = os.path.join(FIG_DIR, 'comprehensive_speedup.png')
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Salvo: {path}")



def generate_summary_table() -> None:
    csv_path = os.path.join(CSV_DIR, 'benchmark_results.csv')
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    df = df[df['time_seconds'] != 'NA']
    df['time_seconds'] = df['time_seconds'].astype(float)

    stats = df.groupby('version')['time_seconds'].agg(['mean', 'std', 'min', 'max']).reset_index()

    seq_time = stats[stats['version'] == 'sequential']['mean'].values
    if len(seq_time) > 0:
        stats['speedup'] = seq_time[0] / stats['mean']
    else:
        stats['speedup'] = float('nan')

    stats.columns = ['Versao', 'Media (s)', 'Desvio Padrao', 'Min (s)', 'Max (s)', 'Speedup']

    summary_path = os.path.join(CSV_DIR, 'summary_table.csv')
    stats.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"  Salvo: {summary_path}")
    print()
    print(stats.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


def main() -> None:
    print("=== Analise de Resultados K-means ===\n")
    setup()

    print("1. Tabela resumo:")
    generate_summary_table()

    print("\n2. Comparacao de benchmark:")
    plot_benchmark_comparison()

    print("\n3. Escalabilidade forte:")
    plot_strong_scaling()

    print("\n4. Escalabilidade fraca:")
    plot_weak_scaling()

    print("\n5. Benchmark Compreensivo (MPI vs CUDA vs Seq):")
    plot_comprehensive_comparison()

    print("\n=== Analise concluida ===")
    print(f"Graficos salvos em: {FIG_DIR}")


if __name__ == '__main__':
    main()
