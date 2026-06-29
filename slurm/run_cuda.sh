#!/bin/bash
#SBATCH --job-name=Kmeans_CUDA
#SBATCH --partition=gpu-8-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=results/raw/cuda_%j.out

cd "${SLURM_SUBMIT_DIR:-.}"

echo "=== K-means CUDA ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Data: $(date)"
echo ""

module load compilers/nvidia/cuda/12.6 2>/dev/null || true

nvcc src/4-cuda/kmeans_cuda.cu -o kmeans_cuda -O3

if [ $? -ne 0 ]; then
    echo "ERRO: Falha na compilacao CUDA"
    exit 1
fi

# Variacao do tamanho do dataset (escalabilidade por tamanho de entrada).
# Cada binario recebe: <num_amostras> <caminho_dataset>
SIZES=(7000 14000 35000 70000 140000 280000 560000)
FILES=(data/fashion_mnist_7k.bin data/fashion_mnist_14k.bin data/fashion_mnist_35k.bin \
       data/fashion_mnist_pure.bin data/fashion_mnist_140k.bin data/fashion_mnist_280k.bin \
       data/fashion_mnist_560k.bin)

for i in "${!SIZES[@]}"; do
    N=${SIZES[$i]}
    DATA=${FILES[$i]}
    echo "--- Dataset: $N amostras ($DATA) ---"
    ./kmeans_cuda "$N" "$DATA"
done

rm -f kmeans_cuda
