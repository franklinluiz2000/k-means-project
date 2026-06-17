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

./kmeans_cuda

rm -f kmeans_cuda
