#!/bin/bash
#SBATCH --job-name=Kmeans_OMP_GPU
#SBATCH --partition=gpu-8-v100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=results/raw/omp_gpu_%j.out

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p results/csv

CSV="results/csv/omp_gpu.csv"
echo "version,samples,run,time_seconds,iterations" > "$CSV"

extract_time() {
    grep -oP 'Tempo total de processamento[^:]*:\s*\K[0-9]+\.[0-9]+' "$1" || echo "NA"
}

extract_iters() {
    grep -oP '(Itera..es necess.rias|itera..es)[^:]*:\s*\K[0-9]+' "$1" || echo "NA"
}

RUNS=3

DATASETS=(
    "17500 data/fashion_mnist_17.5k.bin"
    "35000 data/fashion_mnist_35k.bin"
    "70000 data/fashion_mnist_pure.bin"
    "140000 data/fashion_mnist_140k.bin"
    "280000 data/fashion_mnist_280k.bin"
    "560000 data/fashion_mnist_560k.bin"
)

echo "Compilando versões..."
module load compilers/nvidia/nvhpc/24.11 2>/dev/null || true
nvc -mp=gpu -gpu=cc70 -O3 src/3-openmp-gpu/kmeans_openmp_gpu.c -o kmeans_omp_gpu

for ds in "${DATASETS[@]}"; do
    SAMPLES=$(echo $ds | awk '{print $1}')
    FILE=$(echo $ds | awk '{print $2}')

    echo "--- Testando Dataset: $SAMPLES amostras ($FILE) ---"

    for r in $(seq 1 $RUNS); do
        TMPOUT=$(mktemp)
        ./kmeans_omp_gpu "$SAMPLES" "$FILE" > "$TMPOUT" 2>&1
        T=$(extract_time "$TMPOUT")
        IT=$(extract_iters "$TMPOUT")
        echo "openmp_gpu,$SAMPLES,$r,$T,$IT" >> "$CSV"
        rm -f "$TMPOUT"
    done
done

rm -f kmeans_omp_gpu
echo "Benchmark OMP GPU finalizado."
