#!/bin/bash
#SBATCH --job-name=Kmeans_Hibrido
#SBATCH --partition=amd-512           
#SBATCH --nodes=4                     # 4 máquinas (4 Processos MPI)
#SBATCH --ntasks-per-node=1           # 1 Processo MPI por máquina
#SBATCH --cpus-per-task=64            
#SBATCH --time=00:20:00
#SBATCH --output=resultado_kmeans_mpi_openmp_%j.out

# Compila o código habilitando o OpenMP
mpicc -fopenmp kmeans_mpi_openmp.c -o kmeans_mpi_openmp -lm -O3

# Amarra o número de threads do OpenMP ao número de CPUs acima
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Dispara a execução paralela
mpirun ./kmeans_mpi_openmp