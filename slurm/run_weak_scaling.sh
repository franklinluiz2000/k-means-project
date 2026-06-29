#!/bin/bash
#SBATCH --job-name=Kmeans_Weak
#SBATCH --partition=amd-512
#SBATCH --nodes=1                     # UM unico no
#SBATCH --ntasks=1                    # alocacao flexivel; mpirun controla os ranks
#SBATCH --cpus-per-task=64            # 64 cores reservados no no
#SBATCH --time=02:00:00
#SBATCH --output=results/raw/weak_scaling_%j.out

# ===========================================================================
# ESCALABILIDADE FRACA (Weak Scaling) -- 1 no
# Carga FIXA por processo MPI = 17.500 amostras (threads OpenMP fixo = 4).
# N cresce junto com o numero de processos:  N = 17.500 x tasks.
# Idealmente o tempo permanece CONSTANTE; Eficiencia = T(1) / T(tasks).
# CSV: total_cores,processos,threads_por_processo,amostras,amostras_por_processo,execucao,tempo_seg,iteracoes
# ===========================================================================

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results/csv results/raw

CORES=64
THREADS=4                 # threads OpenMP por processo (fixo)
PER_TASK=17500            # carga por processo MPI (fixa)
RUNS=3

echo "=== Escalabilidade Fraca em 1 nó (~$PER_TASK amostras por processo) ==="
echo "Threads OpenMP por processo: $THREADS | Repeticoes: $RUNS | Cores no nó: $CORES"
echo ""

CSV="results/csv/weak_scaling.csv"
echo "total_cores,processos,threads_por_processo,amostras,amostras_por_processo,execucao,tempo_seg,iteracoes" > "$CSV"

extract_time() {
    grep -oP 'Tempo total de processamento[^:]*:\s*\K[0-9]+\.[0-9]+' "$1" || echo "NA"
}
extract_iters() {
    grep -oP '(Itera..es necess.rias|itera..es)[^:]*:\s*\K[0-9]+' "$1" || echo "NA"
}

echo ">>> Compilando MPI+OpenMP..."
mpicc src/2-paralell-mpi-openmp/kmeans_mpi_openmp.c -o kmeans_mpi_omp -lm -O3 -fopenmp
if [ $? -ne 0 ]; then echo "ERRO: Falha na compilacao"; exit 1; fi
echo ""

export OMP_NUM_THREADS=$THREADS

# Cada entrada: "tasks samples arquivo" (samples = 17500 * tasks)
COMBOS=(
    "1  17500  data/fashion_mnist_17.5k.bin"
    "2  35000  data/fashion_mnist_35k.bin"
    "4  70000  data/fashion_mnist_pure.bin"
    "8  140000 data/fashion_mnist_140k.bin"
    "16 280000 data/fashion_mnist_280k.bin"
)

for combo in "${COMBOS[@]}"; do
    TASKS=$(echo $combo | awk '{print $1}')
    SAMPLES=$(echo $combo | awk '{print $2}')
    FILE=$(echo $combo | awk '{print $3}')
    TOTAL=$((TASKS * THREADS))

    [ "$TOTAL" -gt "$CORES" ] && { echo "PULANDO $TASKS x $THREADS = $TOTAL > $CORES cores"; continue; }

    echo ">>> $TASKS processo(s) x $THREADS threads = $TOTAL cores | $SAMPLES amostras ($FILE)"

    for r in $(seq 1 $RUNS); do
        TMPOUT=$(mktemp)
        mpirun --bind-to none --oversubscribe -np "$TASKS" \
            ./kmeans_mpi_omp "$SAMPLES" "$FILE" > "$TMPOUT" 2>&1
        T=$(extract_time "$TMPOUT")
        IT=$(extract_iters "$TMPOUT")
        echo "   run $r: ${T}s"
        echo "$TOTAL,$TASKS,$THREADS,$SAMPLES,$PER_TASK,$r,$T,$IT" >> "$CSV"
        rm -f "$TMPOUT"
    done
    echo ""
done

rm -f kmeans_mpi_omp
echo "=== Escalabilidade Fraca (1 nó) concluida ==="
echo "Resultados em: $CSV"
