"""
Gera os graficos de Escalabilidade Fraca do K-means (MPI x OpenMP).

Le results/csv/weak_scaling.csv no formato:
    total_threads,processos,threads_por_processo,amostras,amostras_por_processo,
    execucao,tempo_seg,iteracoes

Carga fixa por processo (amostras_por_processo constante); o total de amostras
cresce junto com o numero de unidades de processamento. A eficiencia fraca usa
o primeiro ponto T(1) como base.

Cada ponto e anotado com a quantidade TOTAL de amostras daquele caso.

Gera em results/figures/weak/:
    weak_a_tempo.png       - tempo total vs total_threads
    weak_b_tempo_iter.png  - tempo por iteracao vs total_threads
    weak_c_eficiencia.png  - eficiencia (tempo bruto e por iteracao)
    weak_d_iteracoes.png   - iteracoes ate convergir

Uso: python results/plot_weak_scaling.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_PATH = os.path.join(PROJECT_DIR, 'results', 'csv', 'weak_scaling.csv')
FIG_DIR = os.path.join(PROJECT_DIR, 'results', 'figures', 'weak')


def _fmt_amostras(n: int) -> str:
    return f"{int(n):,}".replace(',', '.')


def load_stats() -> tuple[pd.DataFrame, int]:
    """Le o CSV e devolve (stats_por_config ordenado, carga_por_processo)."""
    df = pd.read_csv(CSV_PATH)
    df = df[df['tempo_seg'] != 'NA']
    df['tempo_seg'] = df['tempo_seg'].astype(float)

    carga = int(df['amostras_por_processo'].iloc[0])

    stats = (
        df.groupby(['total_threads', 'processos', 'threads_por_processo', 'amostras'])
        .agg(tempo=('tempo_seg', 'mean'),
             desvio=('tempo_seg', 'std'),
             iteracoes=('iteracoes', 'mean'))
        .reset_index()
        .sort_values('total_threads')
        .reset_index(drop=True)
    )
    stats['desvio'] = stats['desvio'].fillna(0.0)
    stats['tempo_iter'] = stats['tempo'] / stats['iteracoes']

    base_t = stats['tempo'].iloc[0]
    base_ti = stats['tempo_iter'].iloc[0]
    stats['eff_bruto'] = base_t / stats['tempo'] * 100
    stats['eff_iter'] = base_ti / stats['tempo_iter'] * 100

    stats['xlabel'] = (stats['total_threads'].astype(str) + '\n('
                       + stats['processos'].astype(str) + 'x'
                       + stats['threads_por_processo'].astype(str) + ')')
    return stats, carga


def _annotate_amostras(ax, xs, ys, stats, dy_frac=0.04) -> None:
    """Escreve o total de amostras acima de cada ponto."""
    ymin, ymax = ax.get_ylim()
    dy = (ymax - ymin) * dy_frac
    for x, y, n in zip(xs, ys, stats['amostras']):
        ax.annotate(_fmt_amostras(n), (x, y + dy), ha='center', va='bottom',
                    fontsize=9, color='#333',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))


def _xaxis(ax, stats) -> None:
    ax.set_xticks(stats['total_threads'])
    ax.set_xticklabels(stats['xlabel'], fontsize=9)
    ax.set_xlabel('total_threads (proc x thread)')


def plot_a_tempo(stats: pd.DataFrame, carga: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = stats['total_threads']
    y = stats['tempo']
    ax.errorbar(x, y, yerr=stats['desvio'], fmt='o-', color='#d62728',
                linewidth=2, markersize=8, capsize=4)
    ax.margins(y=0.18)
    _annotate_amostras(ax, x, y, stats)
    _xaxis(ax, stats)
    ax.set_ylabel('Tempo (s)')
    ax.set_title(f'Escalabilidade Fraca - Tempo total (carga fixa {_fmt_amostras(carga)}/proc)')
    _save(fig, 'weak_a_tempo.png')


def plot_b_tempo_iter(stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = stats['total_threads']
    y = stats['tempo_iter']
    ax.plot(x, y, 's-', color='#1f77b4', linewidth=2, markersize=8)
    ax.margins(y=0.18)
    _annotate_amostras(ax, x, y, stats)
    _xaxis(ax, stats)
    ax.set_ylabel('Tempo / iteracao (s)')
    ax.set_title('Escalabilidade Fraca - Tempo POR ITERACAO')
    _save(fig, 'weak_b_tempo_iter.png')


def plot_c_eficiencia(stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = stats['total_threads']
    ax.plot(x, stats['eff_bruto'], 'o-', color='#d62728', linewidth=2,
            markersize=8, label='tempo bruto')
    ax.plot(x, stats['eff_iter'], 's-', color='#1f77b4', linewidth=2,
            markersize=8, label='por iteracao')
    ax.axhline(y=100, color='#2ca02c', linestyle='--', linewidth=1.5, label='ideal (100%)')
    ax.set_ylim(0, 115)
    _annotate_amostras(ax, x, stats['eff_bruto'], stats, dy_frac=0.03)
    _xaxis(ax, stats)
    ax.set_ylabel('Eficiencia (%)')
    ax.set_title('Escalabilidade Fraca - Eficiencia  E=T(1)/T(P)')
    ax.legend(loc='lower left')
    _save(fig, 'weak_c_eficiencia.png')


def plot_d_iteracoes(stats: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = stats['total_threads']
    y = stats['iteracoes']
    ax.plot(x, y, '^-', color='#9467bd', linewidth=2, markersize=9)
    ax.margins(y=0.18)
    _annotate_amostras(ax, x, y, stats)
    _xaxis(ax, stats)
    ax.set_ylabel('No de iteracoes')
    ax.set_title('Escalabilidade Fraca - Iteracoes ate convergir')
    _save(fig, 'weak_d_iteracoes.png')


def _save(fig, name: str) -> None:
    plt.tight_layout()
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})

    stats, carga = load_stats()
    print(f"=== Escalabilidade Fraca (carga={carga}/proc) ===")
    plot_a_tempo(stats, carga)
    plot_b_tempo_iter(stats)
    plot_c_eficiencia(stats)
    plot_d_iteracoes(stats)
    print("=== Concluido ===")


if __name__ == '__main__':
    main()
