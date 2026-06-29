# Implementação e Análise de Desempenho do Algoritmo K-means Paralelo

Projeto prático da 3ª unidade da disciplina IMD1116 - Computação de Alto Desempenho (UFRN/IMD), ministrada pelos professores Prof. Samuel Xavier de Souza e Prof. Carla Santana.

O objetivo é implementar o algoritmo K-means em quatro versões (sequencial, MPI+OpenMP, OpenMP GPU, CUDA), executar no cluster NPAD e comparar speedup, eficiência e escalabilidade.

## Equipe e Divisão de Tarefas

| Integrante | Responsabilidade |
|------------|-----------------|
| Franklin Luiz da Cruz | Implementação Sequencial (Baseline) e Análise Teórica |
| Daniel Vitor de Oliveira Bezerra | Arquitetura e Implementação Híbrida CPU (MPI + OpenMP) |
| Raimundo Marciano de Freitas Neto | Paralelismo em GPU usando OpenMP (offloading) |
| Luiz Gonzaga Gomes Neto | Programação Nativa em GPU usando CUDA |
| Luiz Gustavo de Souza Rego | Engenharia de Dados, Automação de Testes e Scripts no Cluster |

## Dataset

**Fashion MNIST** — 70.000 imagens de roupas (28x28 pixels), achatadas em vetores de 784 dimensões, normalizadas entre 0.0 e 1.0. Armazenado em formato binário (float64) para leitura rápida via `fread()`.

## Versões Desenvolvidas

| # | Versão | Paradigma | Hardware |
|---|--------|-----------|----------|
| 1 | Sequencial | Nenhum (baseline) | 1 core de CPU |
| 2 | MPI + OpenMP | Memória distribuída + compartilhada | Múltiplos cores/máquinas |
| 3 | OpenMP GPU | Offloading com diretivas | GPU |
| 4 | CUDA | Programação nativa de GPU | GPU NVIDIA |

## Estrutura do Projeto

```
k-means-project/
├── src/
│   ├── 1-sequential/kmeans_sequential.c           # [Franklin]
│   ├── 2-paralell-mpi-openmp/kmeans_mpi_openmp.c  # [Daniel]
│   ├── 3-openmp-gpu/kmeans_openmp_gpu.c           # [Raimundo]
│   └── 4-cuda/kmeans_cuda.cu                      # [Luiz Gonzaga]
├── scripts/                                        # [Luiz Gustavo] — apenas Python
│   ├── data_engineering.py      # Download e pré-processamento do Fashion MNIST
│   ├── generate_datasets.py     # Gera datasets de 17.5k a 560k para escalabilidade
│   ├── validate_data.py         # Valida integridade dos binários
│   └── visualize_centroids.py   # Compara visualmente os centroides das versões
├── slurm/                                          # [Luiz Gustavo] — runners SLURM
│   ├── run_sequential.sh        # Job SLURM: sequencial
│   ├── run_mpi_openmp.sh        # Job SLURM: MPI+OpenMP (parametrizável)
│   ├── run_openmp_gpu.sh        # Job SLURM: OpenMP GPU
│   ├── run_cuda.sh              # Job SLURM: CUDA
│   ├── run_strong_scaling.sh    # Escalabilidade forte (N fixo = 70k, varia proc×thread)
│   └── run_weak_scaling.sh      # Escalabilidade fraca (carga fixa 17.5k/processo)
├── data/                        # Datasets binários (ignorado pelo git)
└── results/                                         # Análise e gráficos (Python)
    ├── analyze_results.py       # Gráficos gerais (benchmark, comparações)
    ├── plot_strong_scaling.py   # Gráficos de escalabilidade forte (strong/)
    ├── plot_weak_scaling.py     # Gráficos de escalabilidade fraca (weak/)
    ├── plot_overview.py         # Dashboard de análise geral das 4 versões (overview/)
    ├── plot_gpu.py              # Gráficos específicos de GPU
    ├── raw/                     # Outputs do SLURM e centroides .bin
    ├── csv/                     # Dados processados (strong_scaling.csv, weak_scaling.csv, ...)
    └── figures/                 # Gráficos gerados (strong/, weak/, overview/)
```

## Como Executar

### 1. Preparar os dados

```bash
python3 scripts/data_engineering.py    # Baixa e processa o Fashion MNIST
python3 scripts/generate_datasets.py   # Gera datasets para escalabilidade
python3 scripts/validate_data.py       # Valida integridade
```

### 2. Executar no cluster NPAD

```bash
cd k-means-project
# Uma execução por versão
sbatch slurm/run_sequential.sh         # Sequencial (baseline)
sbatch slurm/run_mpi_openmp.sh         # MPI + OpenMP (parametrizável)
sbatch slurm/run_openmp_gpu.sh         # OpenMP GPU
sbatch slurm/run_cuda.sh               # CUDA
# Experimentos de escalabilidade (MPI + OpenMP)
sbatch slurm/run_strong_scaling.sh     # Escalabilidade forte (N fixo = 70k)
sbatch slurm/run_weak_scaling.sh       # Escalabilidade fraca (carga fixa/processo)
squeue -u $USER                        # Acompanhar jobs
```

### 3. Gerar gráficos

```bash
python3 results/analyze_results.py        # Gráficos gerais
python3 results/plot_strong_scaling.py    # Escalabilidade forte  -> results/figures/strong/
python3 results/plot_weak_scaling.py      # Escalabilidade fraca  -> results/figures/weak/
python3 results/plot_overview.py          # Análise geral das 4 versões -> results/figures/overview/
# Gráficos salvos em results/figures/
```

## Métricas Analisadas

* Tempo de Execução
* Speedup
* Eficiência
* Escalabilidade Forte e Fraca
* Overhead de comunicação e transferência CPU-GPU
