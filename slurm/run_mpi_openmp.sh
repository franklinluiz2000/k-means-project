#!/bin/bash
#SBATCH --job-name=Kmeans_MPI_OMP
#SBATCH --partition=amd-512
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=results/raw/mpi_openmp_%j.out

# Parametros configuraveis via linha de comando:
#   sbatch --nodes=2 --cpus-per-task=16 slurm/run_mpi_openmp.sh

cd "${SLURM_SUBMIT_DIR:-.}"

echo "=== K-means Hibrido (MPI + OpenMP) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Threads/task: $SLURM_CPUS_PER_TASK"
echo "Node list: $SLURM_NODELIST"
echo "Data: $(date)"
echo ""

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpicc src/2-paralell-mpi-openmp/kmeans_mpi_openmp.c -o kmeans_mpi_omp -lm -O3 -fopenmp

if [ $? -ne 0 ]; then
    echo "ERRO: Falha na compilacao"
    exit 1
fi

mpirun ./kmeans_mpi_omp

rm -f kmeans_mpi_omp
