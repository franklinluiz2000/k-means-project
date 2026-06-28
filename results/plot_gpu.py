"""Graficos PNG dos resultados de K-means a partir de gpu_comprehensive.csv.

Gera duas categorias de figuras:
  - juntos/    : todas as versoes sobrepostas no mesmo grafico
  - separados/ : uma figura por versao / comparacao direta

Limitado a datasets ate MAX_SAMPLES. So usa metricas suportadas pelos dados
existentes (tempo, speedup, OpenMP-GPU vs CUDA). Escalabilidade forte/fraca,
eficiencia e overhead MPI exigem CSVs que nao existem no projeto.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

MAX_SAMPLES = 560000

BASE = os.path.dirname(__file__)
CSV = os.path.join(BASE, "csv", "gpu_comprehensive.csv")
OUT_JUNTOS = os.path.join(BASE, "graficos", "juntos")
OUT_SEPAR = os.path.join(BASE, "graficos", "separados")
os.makedirs(OUT_JUNTOS, exist_ok=True)
os.makedirs(OUT_SEPAR, exist_ok=True)

df = pd.read_csv(CSV)
df = df[df["samples"] <= MAX_SAMPLES]

pivot = (
    df.groupby(["version", "samples"], as_index=False)["time_seconds"].mean()
    .pivot(index="samples", columns="version", values="time_seconds")[
        ["sequential", "openmp_gpu", "cuda"]
    ]
)

speedup = pivot.copy()
for col in ["openmp_gpu", "cuda"]:
    speedup[col] = pivot["sequential"] / pivot[col]

sizes = pivot.index.tolist()

STYLE = {
    "sequential": {"color": "#3498db", "marker": "o", "label": "Sequencial (CPU)"},
    "openmp_gpu": {"color": "#e67e22", "marker": "s", "label": "OpenMP-GPU"},
    "cuda": {"color": "#e74c3c", "marker": "^", "label": "CUDA"},
}

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 130, "savefig.dpi": 130})


def kfmt(x, _):
    return f"{int(x/1000)}k"


def style_x(ax):
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(kfmt))
    ax.set_xticks(sizes)
    ax.set_xlabel("Numero de amostras")


def save(fig, folder, name):
    fig.tight_layout()
    path = os.path.join(folder, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Salvo: {os.path.relpath(path, BASE)}")


# ============================ JUNTOS ========================================
# J1) Tempo de execucao - todas as versoes
fig, ax = plt.subplots(figsize=(9, 5.5))
for col, st in STYLE.items():
    ax.plot(sizes, pivot[col], marker=st["marker"], color=st["color"],
            label=st["label"], linewidth=2, markersize=7)
ax.set_ylabel("Tempo (s)")
ax.set_title("Tempo de Execucao por Tamanho do Problema")
style_x(ax)
ax.legend()
save(fig, OUT_JUNTOS, "01_tempo_todas.png")

# J2) Speedup - GPU vs sequencial
fig, ax = plt.subplots(figsize=(9, 5.5))
for col in ["openmp_gpu", "cuda"]:
    st = STYLE[col]
    ax.plot(sizes, speedup[col], marker=st["marker"], color=st["color"],
            label=st["label"], linewidth=2, markersize=7)
    for x, y in zip(sizes, speedup[col]):
        ax.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
ax.set_ylabel("Speedup (x) vs Sequencial")
ax.set_title("Speedup da GPU sobre a CPU Sequencial")
style_x(ax)
ax.legend()
save(fig, OUT_JUNTOS, "02_speedup_todas.png")

# J3) OpenMP-GPU vs CUDA - tempo (linear), comparacao direta
fig, ax = plt.subplots(figsize=(9, 5.5))
for col in ["openmp_gpu", "cuda"]:
    st = STYLE[col]
    ax.plot(sizes, pivot[col], marker=st["marker"], color=st["color"],
            label=st["label"], linewidth=2, markersize=7)
ax.set_ylabel("Tempo (s)")
ax.set_title("Comparacao de Desempenho: OpenMP-GPU vs CUDA")
style_x(ax)
ax.legend()
save(fig, OUT_JUNTOS, "03_openmpgpu_vs_cuda_tempo.png")

# J4) Barras agrupadas de tempo por tamanho (todas as versoes GPU)
fig, ax = plt.subplots(figsize=(10, 5.5))
import numpy as np
x = np.arange(len(sizes))
w = 0.38
ax.bar(x - w/2, pivot["openmp_gpu"], w, color=STYLE["openmp_gpu"]["color"],
       label="OpenMP-GPU")
ax.bar(x + w/2, pivot["cuda"], w, color=STYLE["cuda"]["color"], label="CUDA")
ax.set_xticks(x)
ax.set_xticklabels([f"{int(s/1000)}k" for s in sizes])
ax.set_xlabel("Numero de amostras")
ax.set_ylabel("Tempo (s)")
ax.set_title("Tempo por Tamanho: OpenMP-GPU vs CUDA")
ax.grid(axis="x")
ax.legend()
save(fig, OUT_JUNTOS, "04_barras_tempo_gpu.png")

# ============================ SEPARADOS =====================================
# S1-S3) Tempo de execucao por versao (individual)
for col in ["sequential", "openmp_gpu", "cuda"]:
    st = STYLE[col]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, pivot[col], marker=st["marker"], color=st["color"],
            linewidth=2, markersize=7)
    for xv, yv in zip(sizes, pivot[col]):
        ax.annotate(f"{yv:.2f}s", (xv, yv), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_ylabel("Tempo (s)")
    ax.set_title(f"Tempo de Execucao - {st['label']}")
    style_x(ax)
    save(fig, OUT_SEPAR, f"tempo_{col}.png")

# S4-S5) Speedup por versao (individual)
for col in ["openmp_gpu", "cuda"]:
    st = STYLE[col]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, speedup[col], marker=st["marker"], color=st["color"],
            linewidth=2, markersize=7)
    for xv, yv in zip(sizes, speedup[col]):
        ax.annotate(f"{yv:.1f}x", (xv, yv), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_ylabel("Speedup (x) vs Sequencial")
    ax.set_title(f"Speedup - {st['label']}")
    style_x(ax)
    save(fig, OUT_SEPAR, f"speedup_{col}.png")

# S6) Barras de speedup no maior problema considerado
fig, ax = plt.subplots(figsize=(7, 5))
biggest = sizes[-1]
vals = [speedup.loc[biggest, "openmp_gpu"], speedup.loc[biggest, "cuda"]]
bars = ax.bar(["OpenMP-GPU", "CUDA"], vals,
              color=[STYLE["openmp_gpu"]["color"], STYLE["cuda"]["color"]], width=0.55)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:.1f}x", (b.get_x() + b.get_width()/2, v),
                textcoords="offset points", xytext=(0, 5), ha="center",
                fontweight="bold")
ax.set_ylabel("Speedup (x) vs Sequencial")
ax.set_title(f"Speedup no maior problema ({int(biggest/1000)}k amostras)")
ax.grid(axis="x")
save(fig, OUT_SEPAR, "speedup_barras_maior.png")

# ============================ METRICAS EXTRA ================================
import numpy as np

g = df.groupby(["version", "samples"])
std = g["time_seconds"].std().unstack(0)[["sequential", "openmp_gpu", "cuda"]]
iters = g["iterations"].mean().unstack(0)[["sequential", "openmp_gpu", "cuda"]]
time_per_iter = pivot / iters
ratio = pivot["cuda"] / pivot["openmp_gpu"]  # quantas vezes OpenMP-GPU e mais rapido

# --- #1 Tempo medio com barras de erro (juntos) ---
fig, ax = plt.subplots(figsize=(9, 5.5))
for col, st in STYLE.items():
    ax.errorbar(sizes, pivot[col], yerr=std[col], fmt=st["marker"] + "-",
                color=st["color"], label=st["label"], linewidth=2,
                markersize=7, capsize=5)
ax.set_ylabel("Tempo (s)")
ax.set_title("Tempo Medio com Desvio das 3 Execucoes")
style_x(ax)
ax.legend()
save(fig, OUT_JUNTOS, "05_tempo_barras_erro.png")

# --- #1 separado: coeficiente de variacao (%) por versao ---
cv = (std / pivot * 100)
fig, ax = plt.subplots(figsize=(8, 5))
for col, st in STYLE.items():
    ax.plot(sizes, cv[col], marker=st["marker"], color=st["color"],
            label=st["label"], linewidth=2, markersize=7)
ax.set_ylabel("Coef. de variacao (%)")
ax.set_title("Variabilidade das Medicoes (desvio / media)")
style_x(ax)
ax.legend()
save(fig, OUT_SEPAR, "variabilidade_cv.png")

# --- #2 Tempo por iteracao (juntos) ---
fig, ax = plt.subplots(figsize=(9, 5.5))
for col, st in STYLE.items():
    ax.plot(sizes, time_per_iter[col], marker=st["marker"], color=st["color"],
            label=st["label"], linewidth=2, markersize=7)
ax.set_ylabel("Tempo por iteracao (s)")
ax.set_title("Custo de Uma Iteracao do K-means")
style_x(ax)
ax.legend()
save(fig, OUT_JUNTOS, "06_tempo_por_iteracao.png")

# --- #2 separado: tempo por iteracao so GPU (linear) ---
fig, ax = plt.subplots(figsize=(8, 5))
for col in ["openmp_gpu", "cuda"]:
    st = STYLE[col]
    ax.plot(sizes, time_per_iter[col], marker=st["marker"], color=st["color"],
            label=st["label"], linewidth=2, markersize=7)
ax.set_ylabel("Tempo por iteracao (s)")
ax.set_title("Custo por Iteracao - OpenMP-GPU vs CUDA")
style_x(ax)
ax.legend()
save(fig, OUT_SEPAR, "tempo_por_iteracao_gpu.png")

# --- #4 Razao OpenMP-GPU vs CUDA (separado) ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sizes, ratio, marker="D", color="#8e44ad", linewidth=2, markersize=8)
for xv, yv in zip(sizes, ratio):
    ax.annotate(f"{yv:.1f}x", (xv, yv), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9)
ax.axhline(1, color="#95a5a6", linestyle="--", linewidth=1,
           label="Empate (1x)")
ax.set_ylabel("t(CUDA) / t(OpenMP-GPU)")
ax.set_title("Quantas Vezes o OpenMP-GPU e Mais Rapido que o CUDA")
style_x(ax)
ax.legend()
save(fig, OUT_SEPAR, "razao_openmpgpu_vs_cuda.png")

print("\nTempo por iteracao (s):")
print(time_per_iter.round(4).to_string())
print("\nRazao t(CUDA)/t(OpenMP-GPU):")
print(ratio.round(2).to_string())

print("\nTempos medios (s):")
print(pivot.round(3).to_string())
print("\nSpeedup (x):")
print(speedup[["openmp_gpu", "cuda"]].round(2).to_string())
print(f"\nLimite de amostras: {MAX_SAMPLES}")
