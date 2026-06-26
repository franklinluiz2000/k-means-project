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
├── scripts/                                        # [Luiz Gustavo]
│   ├── data_engineering.py      # Download e pré-processamento do Fashion MNIST
│   ├── generate_datasets.py     # Gera datasets de 7k a 560k para escalabilidade
│   ├── validate_data.py         # Valida integridade dos binários
│   ├── run_benchmarks.sh        # Benchmark automatizado (5 repetições, CSV)
│   ├── test_correctness.sh      # Valida paralelo == sequencial
│   ├── run_strong_scaling.sh    # Escalabilidade forte (problema fixo)
│   ├── run_weak_scaling.sh      # Escalabilidade fraca (dados proporcionais)
│   └── analyze_results.py       # Gera gráficos com matplotlib
├── slurm/                                          # [Luiz Gustavo]
│   ├── run_sequential.sh        # Job SLURM: sequencial
│   ├── run_mpi_openmp.sh        # Job SLURM: MPI+OpenMP (parametrizável)
│   ├── run_openmp_gpu.sh        # Job SLURM: OpenMP GPU
│   ├── run_cuda.sh              # Job SLURM: CUDA
│   └── run_all.sh               # Submete todos os jobs
├── data/                        # Datasets binários (ignorado pelo git)
└── results/
    ├── raw/                     # Outputs do SLURM
    ├── csv/                     # Dados processados
    └── figures/                 # Gráficos gerados
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
sbatch scripts/run_benchmarks.sh       # Benchmark de todas as versões
sbatch scripts/run_strong_scaling.sh   # Escalabilidade forte
sbatch scripts/run_weak_scaling.sh     # Escalabilidade fraca
squeue -u $USER                        # Acompanhar jobs
```

### 3. Gerar gráficos

```bash
python3 scripts/analyze_results.py
# Gráficos salvos em results/figures/
```

## Métricas Analisadas

* Tempo de Execução
* Speedup
* Eficiência
* Escalabilidade Forte e Fraca
* Overhead de comunicação e transferência CPU-GPU

## Escalabilidade Forte e Fraca

Status de geração dos resultados de escalabilidade por versão. Marcado = resultado já gerado.

| Versão | Escalabilidade Forte | Escalabilidade Fraca | Observação |
|--------|:--------------------:|:--------------------:|------------|
| 1 — Sequencial | ⬜ | ⬜ | Baseline (1 core) — serve de referência para speedup; não escala |
| 2 — MPI + OpenMP | ⬜ | ⬜ | Estudo principal: varia nº de processos (MPI) e threads (OpenMP) |
| 3 — OpenMP-GPU | ⬜ | ⬜ | Escalabilidade por tamanho de entrada; depende da implementação |
| 4 — CUDA | ⬜ | ⬜ | Escalabilidade por tamanho de entrada |

## Checklist para Entrega Final

### Implementações
- [x] Versão 1 — Sequencial (baseline)
- [x] Versão 2 — MPI + OpenMP
- [ ] Versão 3 — OpenMP-GPU (offloading)
- [x] Versão 4 — CUDA

### Experimentos (conforme enunciado)
- [ ] Escalabilidade forte — MPI + OpenMP
- [ ] Escalabilidade fraca — MPI + OpenMP
- [ ] Diferentes tamanhos de entrada na v2, variando nº de processos (MPI) e nº de threads (OpenMP)
- [ ] Comparação de desempenho OpenMP-GPU vs CUDA

### Métricas (para cada experimento)
- [ ] Tempo de execução
- [ ] Speedup
- [ ] Eficiência

### Discussão exigida no relatório
- [ ] Overhead de comunicação no MPI
- [ ] Balanceamento de carga
- [ ] Impacto da transferência de dados entre CPU e GPU