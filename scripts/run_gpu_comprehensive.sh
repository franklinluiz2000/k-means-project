#!/bin/bash
#SBATCH --job-name=Kmeans_Comp_GPU
#SBATCH --partition=gpu-8-v100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=results/raw/comp_gpu_%j.out

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p results/csv

CSV="results/csv/gpu_comprehensive.csv"
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
gcc src/1-sequential/kmeans_sequential.c -o kmeans_seq -lm -O3
module load compilers/nvidia/cuda/12.6 2>/dev/null || true
nvcc src/4-cuda/kmeans_cuda.cu -o kmeans_cuda -O3

for ds in "${DATASETS[@]}"; do
    SAMPLES=$(echo $ds | awk '{print $1}')
    FILE=$(echo $ds | awk '{print $2}')

    echo "--- Testando Dataset: $SAMPLES amostras ($FILE) ---"

    # Sequencial
    for r in $(seq 1 $RUNS); do
        TMPOUT=$(mktemp)
        ./kmeans_seq "$SAMPLES" "$FILE" > "$TMPOUT" 2>&1
        T=$(extract_time "$TMPOUT")
        IT=$(extract_iters "$TMPOUT")
        echo "sequential,$SAMPLES,$r,$T,$IT" >> "$CSV"
        rm -f "$TMPOUT"
    done

    # CUDA
    for r in $(seq 1 $RUNS); do
        TMPOUT=$(mktemp)
        ./kmeans_cuda "$SAMPLES" "$FILE" > "$TMPOUT" 2>&1
        T=$(extract_time "$TMPOUT")
        IT=$(extract_iters "$TMPOUT")
        echo "cuda,$SAMPLES,$r,$T,$IT" >> "$CSV"
        rm -f "$TMPOUT"
    done
done

rm -f kmeans_seq kmeans_cuda
echo "Benchmark GPU finalizado."
