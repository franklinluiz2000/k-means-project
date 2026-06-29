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
    ./kmeans_seq "$N" "$DATA"
done

rm -f kmeans_seq
