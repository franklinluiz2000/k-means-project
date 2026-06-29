#!/bin/bash
#SBATCH --job-name=Kmeans_MPI_OMP
#SBATCH --partition=amd-512
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=results/raw/mpi_openmp_%j.out

# Ajuste:
#   --ntasks-per-node = numero de processos MPI
#   --cpus-per-task   = numero de threads OpenMP por processo
# Via linha de comando (sem editar o arquivo):
#   sbatch --ntasks-per-node=8 --cpus-per-task=4 slurm/run_mpi_openmp.sh

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
    mpirun ./kmeans_mpi_omp "$N" "$DATA"
done

rm -f kmeans_mpi_omp
