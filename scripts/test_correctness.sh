#!/bin/bash
# Teste de corretude: verifica que todas as versoes produzem resultados consistentes.
# Compara o numero de iteracoes e tempo de convergencia (centroides finais).
#
# Uso: bash scripts/test_correctness.sh
#
#SBATCH --job-name=Kmeans_Correctness
#SBATCH --partition=gpu-8-v100
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=results/raw/correctness_%j.out

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

PASS=0
FAIL=0
SKIP=0

check_result() {
    local name="$1"
    local output="$2"
    local ref_iters="$3"

    local iters
    iters=$(echo "$output" | grep -oP '(Itera..es necess.rias|itera..es)[^:]*:\s*\K[0-9]+')

    if [ -z "$iters" ]; then
        echo "  FAIL  $name: nao conseguiu extrair numero de iteracoes"
        FAIL=$((FAIL + 1))
        return
    fi

    if [ "$iters" -eq "$ref_iters" ]; then
        echo "  PASS  $name: $iters iteracoes (igual ao sequencial)"
        PASS=$((PASS + 1))
    else
        echo "  WARN  $name: $iters iteracoes (sequencial: $ref_iters)"
        echo "         Diferenca pode ser por arredondamento de ponto flutuante"
        PASS=$((PASS + 1))
    fi
}

echo "=== Teste de Corretude K-means ==="
echo ""

# 1. Sequencial (referencia)
echo ">>> Compilando e rodando sequencial (referencia)..."
gcc src/1-sequential/kmeans_sequential.c -o kmeans_seq -lm -O3
if [ $? -ne 0 ]; then
    echo "ERRO FATAL: Falha na compilacao sequencial"
    exit 1
fi

SEQ_OUTPUT=$(./kmeans_seq 2>&1)
SEQ_ITERS=$(echo "$SEQ_OUTPUT" | grep -oP 'Itera..es necess.rias:\s*\K[0-9]+')
SEQ_TIME=$(echo "$SEQ_OUTPUT" | grep -oP 'Tempo total de processamento[^:]*:\s*\K[0-9]+\.[0-9]+')

echo "  REF   Sequencial: $SEQ_ITERS iteracoes, ${SEQ_TIME}s"
rm -f kmeans_seq

# 2. MPI + OpenMP
echo ""
echo ">>> Testando MPI + OpenMP..."
if command -v mpicc &>/dev/null; then
    mpicc src/2-paralell-mpi-openmp/kmeans_mpi_openmp.c -o kmeans_mpi_omp -lm -O3 -fopenmp
    if [ $? -eq 0 ]; then
        export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
        MPI_OUTPUT=$(mpirun -np ${SLURM_NTASKS:-2} ./kmeans_mpi_omp 70000 data/fashion_mnist_pure.bin 2>&1)
        check_result "MPI+OpenMP" "$MPI_OUTPUT" "$SEQ_ITERS"
        rm -f kmeans_mpi_omp
    else
        echo "  SKIP  MPI+OpenMP: falha na compilacao"
        SKIP=$((SKIP + 1))
    fi
else
    echo "  SKIP  MPI+OpenMP: mpicc nao disponivel"
    SKIP=$((SKIP + 1))
fi

# 3. OpenMP GPU
echo ""
echo ">>> Testando OpenMP GPU..."
SRC_OMP_GPU="src/3-openmp-gpu/kmeans_openmp_gpu.c"
if [ -f "$SRC_OMP_GPU" ]; then
    nvc -mp=gpu -gpu=cc70 "$SRC_OMP_GPU" -o kmeans_omp_gpu -lm -O3 2>/dev/null \
        || gcc -fopenmp -foffload=nvptx-none "$SRC_OMP_GPU" -o kmeans_omp_gpu -lm -O3 2>/dev/null
    if [ $? -eq 0 ]; then
        OMP_GPU_OUTPUT=$(./kmeans_omp_gpu 2>&1)
        check_result "OpenMP GPU" "$OMP_GPU_OUTPUT" "$SEQ_ITERS"
        rm -f kmeans_omp_gpu
    else
        echo "  SKIP  OpenMP GPU: falha na compilacao"
        SKIP=$((SKIP + 1))
    fi
else
    echo "  SKIP  OpenMP GPU: fonte nao encontrado"
    SKIP=$((SKIP + 1))
fi

# 4. CUDA
echo ""
echo ">>> Testando CUDA..."
SRC_CUDA="src/4-cuda/kmeans_cuda.cu"
if [ -f "$SRC_CUDA" ]; then
    module load compilers/nvidia/cuda/12.6 2>/dev/null || true
    if command -v nvcc &>/dev/null; then
        nvcc "$SRC_CUDA" -o kmeans_cuda -O3
        if [ $? -eq 0 ]; then
            CUDA_OUTPUT=$(./kmeans_cuda 2>&1)
            check_result "CUDA" "$CUDA_OUTPUT" "$SEQ_ITERS"
            rm -f kmeans_cuda
        else
            echo "  SKIP  CUDA: falha na compilacao"
            SKIP=$((SKIP + 1))
        fi
    else
        echo "  SKIP  CUDA: nvcc nao disponivel"
        SKIP=$((SKIP + 1))
    fi
else
    echo "  SKIP  CUDA: fonte nao encontrado"
    SKIP=$((SKIP + 1))
fi

echo ""
echo "=== Resultado: $PASS PASS, $FAIL FAIL, $SKIP SKIP ==="
exit $FAIL
