#!/bin/bash
# Escalabilidade Fraca: dados proporcionais ao numero de processadores.
# Cada processador sempre processa ~17.500 amostras.
#
# Pre-requisito: gerar datasets com python scripts/generate_datasets.py
#
# Uso: sbatch scripts/run_weak_scaling.sh
#
#SBATCH --job-name=Kmeans_WeakScale
#SBATCH --partition=amd-512
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=results/raw/weak_scaling_%j.out

RUNS=3
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

mkdir -p results/csv

CSV="results/csv/weak_scaling.csv"
echo "tasks,samples,samples_per_task,run,time_seconds,iterations" > "$CSV"

extract_time() {
    grep -oP 'Tempo total de processamento[^:]*:\s*\K[0-9]+\.[0-9]+' "$1" || echo "NA"
}

extract_iters() {
    grep -oP '(Itera..es necess.rias|itera..es)[^:]*:\s*\K[0-9]+' "$1" || echo "NA"
}

echo "=== Escalabilidade Fraca ==="
echo "Amostras por processador: ~17.500"
echo "Repeticoes: $RUNS"
echo ""

# Configuracoes: (tasks_MPI, dataset, num_samples)
# Nota: o programa MPI tem num_samples=70000 hardcoded.
# Para escalabilidade fraca completa, sera necessario parametrizar
# o num_samples via argumento de linha de comando.
# Por ora, testamos apenas com o dataset padrao de 70k.
CONFIGS=(
    "1 data/fashion_mnist_17.5k.bin 17500"
    "2 data/fashion_mnist_35k.bin 35000"
    "4 data/fashion_mnist_pure.bin 70000"
    "8 data/fashion_mnist_140k.bin 140000"
)

echo ">>> Compilando MPI+OpenMP..."
mpicc src/2-paralell-mpi-openmp/kmeans_mpi_openmp.c -o kmeans_mpi_omp -lm -O3 -fopenmp

if [ $? -ne 0 ]; then
    echo "ERRO: Falha na compilacao MPI+OpenMP"
    exit 1
fi

export OMP_NUM_THREADS=8

for cfg in "${CONFIGS[@]}"; do
    read -r TASKS DATASET SAMPLES <<< "$cfg"
    SAMPLES_PER_TASK=$((SAMPLES / TASKS))

    if [ ! -f "$DATASET" ]; then
        echo "  SKIP: $DATASET nao encontrado (execute python scripts/generate_datasets.py)"
        continue
    fi

    # 70000 deve ser divisivel pelo numero de tasks
    if [ $((SAMPLES % TASKS)) -ne 0 ]; then
        echo "  SKIP: $SAMPLES nao divisivel por $TASKS tasks"
        continue
    fi

    echo ""
    echo ">>> ${TASKS} task(s), ${SAMPLES} amostras ($DATASET)"

    for r in $(seq 1 $RUNS); do
        TMPOUT=$(mktemp)
        mpirun --bind-to none -np "$TASKS" ./kmeans_mpi_omp "$SAMPLES" "$DATASET" > "$TMPOUT" 2>&1
        T=$(extract_time "$TMPOUT")
        IT=$(extract_iters "$TMPOUT")
        echo "$TASKS,$SAMPLES,$SAMPLES_PER_TASK,$r,$T,$IT" >> "$CSV"
        echo "  Run $r: ${T}s"
        rm -f "$TMPOUT"
    done
done

rm -f kmeans_mpi_omp

echo ""
echo "=== Escalabilidade Fraca concluida ==="
echo "Resultados em: $CSV"
echo ""
echo "NOTA: Para escalabilidade fraca completa, o programa MPI precisa aceitar"
echo "o caminho do dataset e num_samples como argumentos de linha de comando."
echo "Solicitar ao Daniel a parametrizacao do kmeans_mpi_openmp.c."
