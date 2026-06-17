#!/bin/bash
#SBATCH --job-name=Kmeans_Comp_MPI
#SBATCH --partition=amd-512
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=results/raw/comp_mpi_%j.out

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p results/csv

CSV="results/csv/mpi_comprehensive.csv"
echo "version,samples,run,time_seconds,iterations" > "$CSV"

extract_time() {
    grep -oP 'Tempo total de processamento[^:]*:\s*\K[0-9]+\.[0-9]+' "$1" || echo "NA"
}

extract_iters() {
    grep -oP '(Itera..es necess.rias|itera..es)[^:]*:\s*\K[0-9]+' "$1" || echo "NA"
}

RUNS=3
TOTAL_TASKS=$SLURM_NTASKS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATASETS=(
    "17500 data/fashion_mnist_17.5k.bin"
    "35000 data/fashion_mnist_35k.bin"
    "70000 data/fashion_mnist_pure.bin"
    "140000 data/fashion_mnist_140k.bin"
    "280000 data/fashion_mnist_280k.bin"
    "560000 data/fashion_mnist_560k.bin"
)

echo "Compilando MPI+OpenMP..."
mpicc src/2-paralell-mpi-openmp/kmeans_mpi_openmp.c -o kmeans_mpi_omp -lm -O3 -fopenmp

for ds in "${DATASETS[@]}"; do
    SAMPLES=$(echo $ds | awk '{print $1}')
    FILE=$(echo $ds | awk '{print $2}')

    echo "--- Testando Dataset: $SAMPLES amostras ($FILE) ---"

    for r in $(seq 1 $RUNS); do
        TMPOUT=$(mktemp)
        mpirun --bind-to none -np "$TOTAL_TASKS" ./kmeans_mpi_omp "$SAMPLES" "$FILE" > "$TMPOUT" 2>&1
        T=$(extract_time "$TMPOUT")
        IT=$(extract_iters "$TMPOUT")
        echo "mpi_openmp_64,$SAMPLES,$r,$T,$IT" >> "$CSV"
        rm -f "$TMPOUT"
    done
done

rm -f kmeans_mpi_omp
echo "Benchmark MPI finalizado."
