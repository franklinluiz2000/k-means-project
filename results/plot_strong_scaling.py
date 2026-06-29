"""
Gera os graficos de Escalabilidade Forte do K-means (MPI x OpenMP).

Le results/csv/strong_scaling.csv no formato:
    total_threads,processos,threads_por_processo,amostras,execucao,tempo_seg,iteracoes

Problema fixo (N amostras), variando processos MPI x threads OpenMP num unico no.
A baseline de speedup/eficiencia e a execucao sequencial (processos=1, threads=1).

Gera em results/figures/strong/:
    strong_a_tempo.png        - tempo por decomposicao (proc x thread)
    strong_b_speedup.png      - speedup por decomposicao
    strong_c_eficiencia.png   - eficiencia por decomposicao
    strong_d_decomposicao.png - curvas tempo vs total_threads por num. de processos
    strong_e_media_desvio.png - tempo medio +/- desvio por total_threads

Uso: python results/plot_strong_scaling.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
CSV_PATH = os.path.join(PROJECT_DIR, 'results', 'csv', 'strong_scaling.csv')
FIG_DIR = os.path.join(PROJECT_DIR, 'results', 'figures', 'strong')

# Paleta por grupo de total_threads (mesma ordem do ciclo padrao do matplotlib)
GROUP_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def load_stats() -> tuple[pd.DataFrame, float, int]:
    """Le o CSV e devolve (stats_por_decomposicao, tempo_seq, n_amostras)."""
    df = pd.read_csv(CSV_PATH)
    df = df[df['tempo_seg'] != 'NA']
    df['tempo_seg'] = df['tempo_seg'].astype(float)

    n_amostras = int(df['amostras'].iloc[0])

    seq = df[(df['processos'] == 1) & (df['threads_por_processo'] == 1)]
    if seq.empty:
        raise SystemExit("ERRO: execucao sequencial (1 proc x 1 thread) ausente no CSV.")
    seq_time = float(seq['tempo_seg'].mean())

    stats = (
        df.groupby(['total_threads', 'processos', 'threads_por_processo'])['tempo_seg']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    stats['std'] = stats['std'].fillna(0.0)
    stats['speedup'] = seq_time / stats['mean']
    stats['efficiency'] = stats['speedup'] / stats['total_threads'] * 100
    stats['decomp'] = (stats['processos'].astype(str) + 'x'
                       + stats['threads_por_processo'].astype(str))

    # Apenas execucoes paralelas (exclui a sequencial 1x1) nos graficos por decomposicao
    par = stats[stats['total_threads'] > 1].copy()
    par = par.sort_values(['total_threads', 'processos']).reset_index(drop=True)
    return par, seq_time, n_amostras


def _group_layout(par: pd.DataFrame):
    """Posicoes de barra com espaco entre grupos de total_threads."""
    groups = sorted(par['total_threads'].unique())
    color_of = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(groups)}
    positions, colors, labels, group_spans = [], [], [], []
    x = 0.0
    for g in groups:
        sub = par[par['total_threads'] == g]
        start = x
        for _, row in sub.iterrows():
            positions.append(x)
            colors.append(color_of[g])
            labels.append(f"{row['processos']}x{row['threads_por_processo']}")
            x += 1.0
        group_spans.append((g, start, x - 1.0))
        x += 1.0  # gap entre grupos
    return positions, colors, labels, group_spans, color_of


def plot_a_tempo(par: pd.DataFrame, n_amostras: int) -> None:
    pos, colors, labels, spans, _ = _group_layout(par)
    times = par['mean'].values

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(pos, times, color=colors, edgecolor='white', linewidth=0.5)
    top = times.max() * 1.12
    for x, t in zip(pos, times):
        ax.text(x, t + top * 0.01, f"{t:.2f}s", ha='center', va='bottom',
                fontsize=8, rotation=90)
    for g, x0, x1 in spans:
        ax.text((x0 + x1) / 2, top * 0.93, f"{g} thr", ha='center',
                fontweight='bold', fontsize=11, color='#444')

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylim(0, top)
    ax.set_ylabel('Tempo (s)')
    ax.set_title(f'Escalabilidade Forte - Tempo por decomposicao (N={n_amostras:,})'
                 .replace(',', '.'))
    _save(fig, 'strong_a_tempo.png')


def plot_b_speedup(par: pd.DataFrame, seq_time: float) -> None:
    pos, colors, labels, spans, color_of = _group_layout(par)
    sp = par['speedup'].values

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(pos, sp, color=colors, edgecolor='white', linewidth=0.5)
    top = sp.max() * 1.12
    for x, s in zip(pos, sp):
        ax.text(x, s + top * 0.01, f"{s:.1f}x", ha='center', va='bottom',
                fontsize=8, rotation=90)
    # Linhas pontilhadas no speedup ideal de cada grupo (= total_threads)
    for g, color in color_of.items():
        ax.axhline(y=g, color=color, linestyle=':', linewidth=1, alpha=0.6)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylim(0, top)
    ax.set_ylabel('Speedup x')
    ax.set_title(f'Escalabilidade Forte - Speedup (base 1thr={seq_time:.0f}s)')
    _save(fig, 'strong_b_speedup.png')


def plot_c_eficiencia(par: pd.DataFrame) -> None:
    pos, colors, labels, spans, _ = _group_layout(par)
    eff = par['efficiency'].values

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(pos, eff, color=colors, edgecolor='white', linewidth=0.5)
    for x, e in zip(pos, eff):
        ax.text(x, e + 1, f"{e:.0f}%", ha='center', va='bottom',
                fontsize=8, rotation=90)
    ax.axhline(y=100, color='#2ca02c', linestyle='--', linewidth=1.5,
               label='ideal (100%)')
    top = max(115, eff.max() * 1.1)
    for g, x0, x1 in spans:
        ax.text((x0 + x1) / 2, top * 0.97, f"{g} thr", ha='center',
                fontweight='bold', fontsize=11, color='#444')

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylim(0, top)
    ax.set_ylabel('Eficiencia (%)')
    ax.set_title('Escalabilidade Forte - Eficiencia por decomposicao')
    ax.legend(loc='upper right')
    _save(fig, 'strong_c_eficiencia.png')


def plot_d_decomposicao(par: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, p in enumerate(sorted(par['processos'].unique())):
        sub = par[par['processos'] == p].sort_values('total_threads')
        ax.plot(sub['total_threads'], sub['mean'], marker='o',
                color=GROUP_COLORS[i % len(GROUP_COLORS)],
                linewidth=2, markersize=8, label=f'{p} proc')

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    xs = sorted(par['total_threads'].unique())
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel('total_threads')
    ax.set_ylabel('Tempo (s)')
    ax.set_title('Escalabilidade Forte - Decomposicao MPI vs OpenMP (curvas ~sobrepostas)')
    ax.legend(title='processos MPI', ncol=2)
    _save(fig, 'strong_d_decomposicao.png')


def plot_e_media_desvio(par: pd.DataFrame, n_amostras: int) -> None:
    agg = (
        par.groupby('total_threads')
        .apply(lambda g: pd.Series({
            'mean': np.average(g['mean'], weights=g['count']),
            'std': g['mean'].std(ddof=0),
            'n': int(g['count'].sum()),
        }), include_groups=False)
        .reset_index()
        .sort_values('total_threads')
    )

    x = np.arange(len(agg))
    colors = [GROUP_COLORS[i % len(GROUP_COLORS)] for i in range(len(agg))]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x, agg['mean'], yerr=agg['std'], color=colors, capsize=4,
           edgecolor='white', linewidth=0.5)
    top = (agg['mean'] + agg['std']).max() * 1.15
    for xi, m, s, n in zip(x, agg['mean'], agg['std'], agg['n']):
        ax.text(xi, m + s + top * 0.01, f"{m:.2f}±{s:.2f}s\n(n={int(n)})",
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(t)) for t in agg['total_threads']])
    ax.set_ylim(0, top)
    ax.set_xlabel('total_threads (proc x thread)')
    ax.set_ylabel('Tempo (s)')
    ax.set_title('Escalabilidade Forte - Tempo medio +/- desvio padrao por total_threads\n'
                 f'(media de todas as decomposicoes procxthread e repeticoes | N={n_amostras:,})'
                 .replace(',', '.'))
    _save(fig, 'strong_e_media_desvio.png')


def _save(fig, name: str) -> None:
    plt.tight_layout()
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Salvo: {path}")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'axes.grid': True, 'grid.alpha': 0.3})

    par, seq_time, n_amostras = load_stats()
    print(f"=== Escalabilidade Forte (N={n_amostras}, base seq={seq_time:.2f}s) ===")
    plot_a_tempo(par, n_amostras)
    plot_b_speedup(par, seq_time)
    plot_c_eficiencia(par)
    plot_d_decomposicao(par)
    plot_e_media_desvio(par, n_amostras)
    print("=== Concluido ===")


if __name__ == '__main__':
    main()
