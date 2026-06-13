#!/bin/bash
#SBATCH --job-name=Kmeans_Seq
#SBATCH --partition=amd-512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=results/raw/sequential_%j.out

cd "${SLURM_SUBMIT_DIR:-.}"

echo "=== K-means Sequencial ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Data: $(date)"
echo ""

gcc src/1-sequential/kmeans_sequential.c -o kmeans_seq -lm -O3

if [ $? -ne 0 ]; then
    echo "ERRO: Falha na compilacao"
    exit 1
fi

./kmeans_seq

rm -f kmeans_seq
