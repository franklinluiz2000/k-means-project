#!/bin/bash
#SBATCH --job-name=Kmeans_Strong
#SBATCH --partition=amd-512
#SBATCH --nodes=1                     # UM unico no
#SBATCH --ntasks=1                    # alocacao flexivel; mpirun controla os ranks
#SBATCH --cpus-per-task=64            # 64 cores reservados no no
#SBATCH --time=02:00:00
#SBATCH --output=results/raw/strong_scaling_%j.out

# ===========================================================================
# ESCALABILIDADE FORTE (Strong Scaling) -- 1 no
# Problema FIXO: N = 70.000 amostras.
# Varre a matriz processos MPI x threads OpenMP, com total = procs*threads <= 64.
# Speedup = T(1,1) / T(procs,threads) ; Eficiencia = Speedup / total_cores.
# CSV: total_cores,processos,threads_por_processo,amostras,execucao,tempo_seg,iteracoes
# ===========================================================================

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results/csv results/raw

CORES=64
SAMPLES=70000
FILE="data/fashion_mnist_pure.bin"
RUNS=3

echo "=== Escalabilidade Forte em 1 nó (processos MPI x threads OpenMP) ==="
echo "Dataset fixo: $SAMPLES amostras | Repeticoes: $RUNS | Cores no nó: $CORES"
echo ""

CSV="results/csv/strong_scaling.csv"
echo "total_cores,processos,threads_por_processo,amostras,execucao,tempo_seg,iteracoes" > "$CSV"

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

PROCS_LIST="1 2 4 8 16"
THREADS_LIST="1 2 4 8 16 32"

for PROCS in $PROCS_LIST; do
    for THREADS in $THREADS_LIST; do
        TOTAL=$((PROCS * THREADS))
        [ "$TOTAL" -gt "$CORES" ] && continue          # nao estoura os 64 cores
        [ $((SAMPLES % PROCS)) -ne 0 ] && continue      # exige divisibilidade MPI

        echo ">>> $PROCS processo(s) MPI x $THREADS thread(s) OpenMP = $TOTAL cores"
        export OMP_NUM_THREADS=$THREADS

        for r in $(seq 1 $RUNS); do
            TMPOUT=$(mktemp)
            mpirun --bind-to none --oversubscribe -np "$PROCS" \
                ./kmeans_mpi_omp "$SAMPLES" "$FILE" > "$TMPOUT" 2>&1
            T=$(extract_time "$TMPOUT")
            IT=$(extract_iters "$TMPOUT")
            echo "   run $r: ${T}s"
            echo "$TOTAL,$PROCS,$THREADS,$SAMPLES,$r,$T,$IT" >> "$CSV"
            rm -f "$TMPOUT"
        done
        echo ""
    done
done

rm -f kmeans_mpi_omp
echo "=== Escalabilidade Forte (1 nó) concluida ==="
echo "Resultados em: $CSV"
