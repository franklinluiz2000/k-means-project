#!/bin/bash
#SBATCH --job-name=Kmeans_OMP_GPU
#SBATCH --partition=gpu-8-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=results/raw/openmp_gpu_%j.out

cd "${SLURM_SUBMIT_DIR:-.}"

echo "=== K-means OpenMP GPU (Offloading) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Data: $(date)"
echo ""

# Compilacao com offloading para GPU (nvc/gcc com suporte a target)
# Ajustar o compilador conforme disponivel no cluster
nvc -mp=gpu -gpu=cc70 src/3-openmp-gpu/kmeans_openmp_gpu.c -o kmeans_omp_gpu -lm -O3 2>/dev/null \
    || gcc -fopenmp -foffload=nvptx-none src/3-openmp-gpu/kmeans_openmp_gpu.c -o kmeans_omp_gpu -lm -O3

if [ $? -ne 0 ]; then
    echo "ERRO: Falha na compilacao (verifique se o compilador suporta offloading GPU)"
    exit 1
fi

./kmeans_omp_gpu

rm -f kmeans_omp_gpu
