"""
Dashboard de Analise Geral - K-means: compara as 4 implementacoes.

Versoes comparadas:
    sequential      - Sequencial (1 CPU)
    mpi_openmp_64   - MPI + OpenMP (64 CPUs)
    openmp_gpu      - OpenMP offload em GPU (1 GPU V100)
    cuda            - CUDA nativo (1 GPU V100)

Le results/csv/gpu_comprehensive.csv e results/csv/mpi_comprehensive.csv
(formato: version,samples,run,time_seconds,iterations) e gera UMA figura com
4 paineis:
    (a) Tempo de execucao vs tamanho do dataset (log-log)
    (b) Speedup vs sequencial
    (c) Throughput (amostras processadas por segundo)
    (d) Speedup no maior dataset comum

Uso: python results/plot_overview.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_DIR = os.path.join(PROJECT_DIR, 'results', 'csv')
FIG_DIR = os.path.join(PROJECT_DIR, 'results', 'figures', 'overview')

ORDER = ['sequential', 'mpi_openmp_64', 'cuda', 'openmp_gpu']
COLORS = {'sequential': '#3498db', 'mpi_openmp_64': '#2ecc71',
          'cuda': '#e74c3c', 'openmp_gpu': '#e67e22'}
MARKERS = {'sequential': 'o', 'mpi_openmp_64': 's', 'cuda': 'D', 'openmp_gpu': '^'}
LABELS = {'sequential': 'Sequencial (1 CPU)',
          'mpi_openmp_64': 'MPI+OpenMP (64 CPUs)',
          'cuda': 'CUDA (1 GPU V100)',
          'openmp_gpu': 'OpenMP-GPU (1 GPU V100)'}


def load_stats() -> pd.DataFrame:
    frames = []
    for name in ('gpu_comprehensive.csv', 'mpi_comprehensive.csv'):
        path = os.path.join(CSV_DIR, name)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("ERRO: nenhum CSV abrangente encontrado.")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=['version', 'samples', 'run'])
    df = df[df['time_seconds'] != 'NA']
    df['time_seconds'] = df['time_seconds'].astype(float)
    df['samples'] = df['samples'].astype(int)

    stats = (df.groupby(['version', 'samples'])['time_seconds']
             .mean().reset_index())
    stats['throughput'] = stats['samples'] / stats['time_seconds']

    seq = stats[stats['version'] == 'sequential'].set_index('samples')['time_seconds']
    stats['speedup'] = stats.apply(
        lambda r: seq[r['samples']] / r['time_seconds']
        if r['samples'] in seq.index else float('nan'), axis=1)
    return stats


def _versions(stats: pd.DataFrame) -> list[str]:
    present = set(stats['version'].unique())
    return [v for v in ORDER if v in present]


def _xticks_samples(ax, stats: pd.DataFrame) -> None:
    xs = sorted(stats['samples'].unique())
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}".replace(',', '.') for x in xs],
                       rotation=30, fontsize=9)
    ax.minorticks_off()


def _save(fig, name: str) -> None:
    plt.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


def plot_tempo(stats: pd.DataFrame, versions: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for v in versions:
        sub = stats[stats['version'] == v].sort_values('samples')
        ax.plot(sub['samples'], sub['time_seconds'], marker=MARKERS[v],
                color=COLORS[v], linewidth=2, markersize=8, label=LABELS[v])
    ax.set_xscale('log'); ax.set_yscale('log')
    _xticks_samples(ax, stats)
    ax.set_xlabel('Amostras'); ax.set_ylabel('Tempo (s) - escala log')
    ax.set_title('Analise Geral - Tempo de execucao vs tamanho do dataset')
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, 'geral_a_tempo.png')


def plot_speedup(stats: pd.DataFrame, versions: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for v in versions:
        if v == 'sequential':
            continue
        sub = stats[stats['version'] == v].sort_values('samples')
        ax.plot(sub['samples'], sub['speedup'], marker=MARKERS[v],
                color=COLORS[v], linewidth=2, markersize=8, label=LABELS[v])
    ax.axhline(y=1, color='#95a5a6', linestyle='--', linewidth=1,
               label='Baseline sequencial (1x)')
    ax.set_xscale('log')
    _xticks_samples(ax, stats)
    ax.set_xlabel('Amostras'); ax.set_ylabel('Speedup (x vs sequencial)')
    ax.set_title('Analise Geral - Speedup em relacao ao sequencial')
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, 'geral_b_speedup.png')


def plot_throughput(stats: pd.DataFrame, versions: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for v in versions:
        sub = stats[stats['version'] == v].sort_values('samples')
        ax.plot(sub['samples'], sub['throughput'], marker=MARKERS[v],
                color=COLORS[v], linewidth=2, markersize=8, label=LABELS[v])
    ax.set_xscale('log'); ax.set_yscale('log')
    _xticks_samples(ax, stats)
    ax.set_xlabel('Amostras'); ax.set_ylabel('Throughput (amostras/s) - escala log')
    ax.set_title('Analise Geral - Vazao (amostras processadas por segundo)')
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, 'geral_c_throughput.png')


def plot_speedup_max(stats: pd.DataFrame, versions: list[str]) -> None:
    par = [v for v in versions if v != 'sequential']
    big = int(stats[stats['version'] == 'sequential']['samples'].max())
    sp_big = []
    for v in par:
        row = stats[(stats['version'] == v) & (stats['samples'] == big)]
        sp_big.append(float(row['speedup'].iloc[0]) if not row.empty else 0.0)

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.bar([LABELS[v] for v in par], sp_big,
                  color=[COLORS[v] for v in par], edgecolor='white')
    for b, s in zip(bars, sp_big):
        ax.text(b.get_x() + b.get_width() / 2, s + max(sp_big) * 0.01,
                f"{s:.1f}x", ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Speedup (x)')
    ax.set_title(f'Analise Geral - Speedup no maior dataset (N={big:,})'.replace(',', '.'))
    ax.tick_params(axis='x', labelrotation=10)
    ax.grid(True, axis='y', alpha=0.3)
    _save(fig, 'geral_d_speedup_max.png')


def main() -> None:
    plt.rcParams.update({'font.size': 12})
    stats = load_stats()
    versions = _versions(stats)
    print("=== Analise Geral (4 implementacoes) ===")
    plot_tempo(stats, versions)
    plot_speedup(stats, versions)
    plot_throughput(stats, versions)
    plot_speedup_max(stats, versions)
    print("=== Concluido ===")


if __name__ == '__main__':
    main()
