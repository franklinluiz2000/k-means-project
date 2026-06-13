#!/bin/bash
# Escalabilidade Forte: problema fixo (70k amostras), varia processadores.
# Testa MPI+OpenMP com diferentes configuracoes de nos e threads.
#
# Uso: sbatch scripts/run_strong_scaling.sh
#
#SBATCH --job-name=Kmeans_StrongScale
#SBATCH --partition=amd-512
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=results/raw/strong_scaling_%j.out

RUNS=3
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

mkdir -p results/csv

CSV="results/csv/strong_scaling.csv"
echo "nodes,tasks,threads_per_task,total_procs,run,time_seconds,iterations" > "$CSV"

extract_time() {
    grep -oP 'Tempo total de processamento[^:]*:\s*\K[0-9]+\.[0-9]+' "$1" || echo "NA"
}

extract_iters() {
    grep -oP '(Itera..es necess.rias|itera..es)[^:]*:\s*\K[0-9]+' "$1" || echo "NA"
}

echo "=== Escalabilidade Forte ==="
echo "Dataset fixo: 70.000 amostras"
echo "Repeticoes: $RUNS"
echo ""

# 1. Baseline sequencial
echo ">>> Sequencial (1 core)..."
gcc src/1-sequential/kmeans_sequential.c -o kmeans_seq -lm -O3
for r in $(seq 1 $RUNS); do
    TMPOUT=$(mktemp)
    ./kmeans_seq > "$TMPOUT" 2>&1
    T=$(extract_time "$TMPOUT")
    IT=$(extract_iters "$TMPOUT")
    echo "1,1,1,1,$r,$T,$IT" >> "$CSV"
    echo "  Run $r: ${T}s"
    rm -f "$TMPOUT"
done
rm -f kmeans_seq

# 2. MPI+OpenMP com diferentes configuracoes
echo ""
echo ">>> Compilando MPI+OpenMP..."
mpicc src/2-paralell-mpi-openmp/kmeans_mpi_openmp.c -o kmeans_mpi_omp -lm -O3 -fopenmp

if [ $? -ne 0 ]; then
    echo "ERRO: Falha na compilacao MPI+OpenMP"
    exit 1
fi

# Configuracoes: (nos, tasks_por_no, threads_por_task)
CONFIGS=(
    "1 1 2"
    "1 1 4"
    "1 1 8"
    "1 1 16"
    "1 1 32"
    "1 1 64"
    "2 1 8"
    "2 1 16"
    "2 1 32"
    "4 1 8"
    "4 1 16"
)

for cfg in "${CONFIGS[@]}"; do
    read -r NODES TASKS_PER_NODE THREADS <<< "$cfg"
    TOTAL_TASKS=$((NODES * TASKS_PER_NODE))
    TOTAL_PROCS=$((TOTAL_TASKS * THREADS))

    # 70000 deve ser divisivel pelo numero de processos MPI
    if [ $((70000 % TOTAL_TASKS)) -ne 0 ]; then
        echo "  SKIP: 70000 nao divisivel por $TOTAL_TASKS tasks"
        continue
    fi

    echo ""
    echo ">>> ${NODES} no(s), ${TOTAL_TASKS} task(s), ${THREADS} threads (total: ${TOTAL_PROCS} procs)"
    export OMP_NUM_THREADS=$THREADS

    for r in $(seq 1 $RUNS); do
        TMPOUT=$(mktemp)
        mpirun -np "$TOTAL_TASKS" ./kmeans_mpi_omp > "$TMPOUT" 2>&1
        T=$(extract_time "$TMPOUT")
        IT=$(extract_iters "$TMPOUT")
        echo "$NODES,$TOTAL_TASKS,$THREADS,$TOTAL_PROCS,$r,$T,$IT" >> "$CSV"
        echo "  Run $r: ${T}s"
        rm -f "$TMPOUT"
    done
done

rm -f kmeans_mpi_omp

echo ""
echo "=== Escalabilidade Forte concluida ==="
echo "Resultados em: $CSV"
