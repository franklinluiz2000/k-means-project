#!/bin/bash
# Benchmark automatizado: roda cada versao N vezes e coleta tempos.
# Uso no cluster: sbatch scripts/run_benchmarks.sh
# Uso local (sequencial apenas): bash scripts/run_benchmarks.sh
#
#SBATCH --job-name=Kmeans_Bench
#SBATCH --partition=gpu-8-v100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=results/raw/benchmark_%j.out

RUNS=5
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

mkdir -p results/csv

CSV="results/csv/benchmark_results.csv"
echo "version,run,time_seconds,iterations" > "$CSV"

extract_time() {
    grep -oP 'Tempo total de processamento[^:]*:\s*\K[0-9]+\.[0-9]+' "$1" || echo "NA"
}

extract_iters() {
    grep -oP '(Itera..es necess.rias|itera..es)[^:]*:\s*\K[0-9]+' "$1" || echo "NA"
}

echo "=== Benchmark K-means ==="
echo "Repeticoes por versao: $RUNS"
echo ""

# --- Sequencial ---
echo ">>> Compilando versao sequencial..."
gcc src/1-sequential/kmeans_sequential.c -o kmeans_seq -lm -O3
if [ $? -eq 0 ]; then
    for i in $(seq 1 $RUNS); do
        echo "  Sequencial - execucao $i/$RUNS"
        TMPOUT=$(mktemp)
        ./kmeans_seq > "$TMPOUT" 2>&1
        T=$(extract_time "$TMPOUT")
        IT=$(extract_iters "$TMPOUT")
        echo "sequential,$i,$T,$IT" >> "$CSV"
        rm -f "$TMPOUT"
    done
    rm -f kmeans_seq
else
    echo "  ERRO: Falha na compilacao sequencial"
fi

# --- MPI + OpenMP ---
echo ""
echo ">>> Compilando versao MPI+OpenMP..."
if command -v mpicc &>/dev/null; then
    mpicc src/2-paralell-mpi-openmp/kmeans_mpi_openmp.c -o kmeans_mpi_omp -lm -O3 -fopenmp
    if [ $? -eq 0 ]; then
        export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
        NTASKS=${SLURM_NTASKS:-4}
        for i in $(seq 1 $RUNS); do
            echo "  MPI+OpenMP - execucao $i/$RUNS (${NTASKS} tasks, ${OMP_NUM_THREADS} threads)"
            TMPOUT=$(mktemp)
            mpirun -np "$NTASKS" ./kmeans_mpi_omp 70000 data/fashion_mnist_pure.bin > "$TMPOUT" 2>&1
            T=$(extract_time "$TMPOUT")
            IT=$(extract_iters "$TMPOUT")
            echo "mpi_openmp,$i,$T,$IT" >> "$CSV"
            rm -f "$TMPOUT"
        done
        rm -f kmeans_mpi_omp
    else
        echo "  ERRO: Falha na compilacao MPI+OpenMP"
    fi
else
    echo "  SKIP: mpicc nao encontrado"
fi

# --- OpenMP GPU ---
echo ""
echo ">>> Compilando versao OpenMP GPU..."
SRC_OMP_GPU="src/3-openmp-gpu/kmeans_openmp_gpu.c"
if [ -f "$SRC_OMP_GPU" ]; then
    nvc -mp=gpu -gpu=cc70 "$SRC_OMP_GPU" -o kmeans_omp_gpu -lm -O3 2>/dev/null \
        || gcc -fopenmp -foffload=nvptx-none "$SRC_OMP_GPU" -o kmeans_omp_gpu -lm -O3 2>/dev/null
    if [ $? -eq 0 ]; then
        for i in $(seq 1 $RUNS); do
            echo "  OpenMP GPU - execucao $i/$RUNS"
            TMPOUT=$(mktemp)
            ./kmeans_omp_gpu > "$TMPOUT" 2>&1
            T=$(extract_time "$TMPOUT")
            IT=$(extract_iters "$TMPOUT")
            echo "openmp_gpu,$i,$T,$IT" >> "$CSV"
            rm -f "$TMPOUT"
        done
        rm -f kmeans_omp_gpu
    else
        echo "  ERRO: Falha na compilacao OpenMP GPU"
    fi
else
    echo "  SKIP: $SRC_OMP_GPU nao encontrado (aguardando Raimundo)"
fi

# --- CUDA ---
echo ""
echo ">>> Compilando versao CUDA..."
SRC_CUDA="src/4-cuda/kmeans_cuda.cu"
if [ -f "$SRC_CUDA" ]; then
    module load compilers/nvidia/cuda/12.6 2>/dev/null || true
    if command -v nvcc &>/dev/null; then
        nvcc "$SRC_CUDA" -o kmeans_cuda -O3
        if [ $? -eq 0 ]; then
            for i in $(seq 1 $RUNS); do
                echo "  CUDA - execucao $i/$RUNS"
                TMPOUT=$(mktemp)
                ./kmeans_cuda > "$TMPOUT" 2>&1
                T=$(extract_time "$TMPOUT")
                IT=$(extract_iters "$TMPOUT")
                echo "cuda,$i,$T,$IT" >> "$CSV"
                rm -f "$TMPOUT"
            done
            rm -f kmeans_cuda
        else
            echo "  ERRO: Falha na compilacao CUDA"
        fi
    else
        echo "  SKIP: nvcc nao encontrado"
    fi
else
    echo "  SKIP: $SRC_CUDA nao encontrado (aguardando Luiz Gonzaga)"
fi

echo ""
echo "=== Benchmark concluido ==="
echo "Resultados salvos em: $CSV"
cat "$CSV"
