#!/bin/bash
# Script mestre: submete todos os jobs K-means ao SLURM.
# Uso: bash slurm/run_all.sh

cd "$(dirname "$0")/.."

echo "=== Submetendo todos os jobs K-means ==="
echo "Data: $(date)"
echo ""

# 1. Sequencial
SEQ_JOB=$(sbatch --parsable slurm/run_sequential.sh)
echo "Sequencial submetido: Job $SEQ_JOB"

# 2. MPI + OpenMP (configuracao padrao: 4 nos, 8 threads)
MPI_JOB=$(sbatch --parsable slurm/run_mpi_openmp.sh)
echo "MPI+OpenMP submetido: Job $MPI_JOB"

# 3. OpenMP GPU
OMP_GPU_JOB=$(sbatch --parsable slurm/run_openmp_gpu.sh)
echo "OpenMP GPU submetido: Job $OMP_GPU_JOB"

# 4. CUDA
CUDA_JOB=$(sbatch --parsable slurm/run_cuda.sh)
echo "CUDA submetido: Job $CUDA_JOB"

echo ""
echo "Todos os jobs submetidos. Use 'squeue -u \$USER' para acompanhar."
echo ""
echo "Jobs:"
echo "  Sequencial:  $SEQ_JOB"
echo "  MPI+OpenMP:  $MPI_JOB"
echo "  OpenMP GPU:  $OMP_GPU_JOB"
echo "  CUDA:        $CUDA_JOB"
