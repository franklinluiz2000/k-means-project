#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <cuda_runtime.h>

#define MAX_ITER 300
#define TOLERANCE 1e-4

#define MAX_K 10
#define MAX_FEATURES 784
__constant__ double d_centroids_const[MAX_K * MAX_FEATURES];

// Macro para debug do CUDA
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void initialize_centroids(const double *dataset, double *centroids, int K, int num_features) {
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < num_features; j++) {
            centroids[i * num_features + j] = dataset[i * num_features + j];
        }
    }
}

// Fallback para atomicAdd com double em arquiteturas mais antigas (se necessário)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__global__ void fused_kmeans_kernel(const double *dataset, 
                                    double *new_centroids, int *cluster_counts,
                                    int num_samples, int num_features, int K) {
    extern __shared__ char smem[];
    double* s_new_centroids = (double*)smem;
    int* s_cluster_counts = (int*)&s_new_centroids[K * num_features];

    // Inicializa a Memória Compartilhada
    for (int i = threadIdx.x; i < K * num_features; i += blockDim.x) {
        s_new_centroids[i] = 0.0;
    }
    if (threadIdx.x < K) {
        s_cluster_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_samples) {
        double dist[MAX_K];
        for (int k = 0; k < K; k++) dist[k] = 0.0;

        for (int j = 0; j < num_features; j++) {
            double val = dataset[j * num_samples + i]; // Transposed coalesced read
            for (int k = 0; k < K; k++) {
                double diff = val - d_centroids_const[k * num_features + j];
                dist[k] += diff * diff;
            }
        }

        double min_dist = DBL_MAX;
        int closest_cluster = 0;
        for (int k = 0; k < K; k++) {
            if (dist[k] < min_dist) {
                min_dist = dist[k];
                closest_cluster = k;
            }
        }
        
        // Atualiza a contagem do cluster NA MEMORIA COMPARTILHADA
        atomicAdd(&s_cluster_counts[closest_cluster], 1);
        
        // Atualiza as dimensões somando os pontos NA MEMORIA COMPARTILHADA
        for (int j = 0; j < num_features; j++) {
            atomicAdd(&s_new_centroids[closest_cluster * num_features + j], dataset[j * num_samples + i]);
        }
    }

    __syncthreads();

    // Redução: Escrita da Memória Compartilhada para a Global apenas 1 vez por bloco
    for (int idx = threadIdx.x; idx < K * num_features; idx += blockDim.x) {
        if (s_new_centroids[idx] != 0.0) {
            atomicAdd(&new_centroids[idx], s_new_centroids[idx]);
        }
    }
    if (threadIdx.x < K) {
        if (s_cluster_counts[threadIdx.x] != 0) {
            atomicAdd(&cluster_counts[threadIdx.x], s_cluster_counts[threadIdx.x]);
        }
    }
}

int main(int argc, char **argv) {
    int num_samples = 70000; 
    int num_features = 784;     
    int K = 10; 
    char dataset_path[256] = "data/fashion_mnist_pure.bin";

    if (argc >= 2) {
        num_samples = atoi(argv[1]);
    }
    if (argc >= 3) {
        snprintf(dataset_path, sizeof(dataset_path), "%s", argv[2]);
    }

    printf("=== HPC K-means CUDA ===\n");
    printf("Dataset: %s\n", dataset_path);
    printf("Configuração: %d amostras, %d dimensões, K=%d\n\n", num_samples, num_features, K);

    // Alocação de Memória no Host
    double *h_dataset = (double *)malloc(num_samples * num_features * sizeof(double));
    double *h_centroids = (double *)malloc(K * num_features * sizeof(double));
    
    if (!h_dataset || !h_centroids) {
        fprintf(stderr, "Erro de alocação de memória no Host.\n");
        return EXIT_FAILURE;
    }

    printf("Carregando dados do arquivo binário...\n");
    FILE *file = fopen(dataset_path, "rb");
    if (!file) {
        fprintf(stderr, "Erro: O arquivo '%s' não foi encontrado!\n", dataset_path);
        free(h_dataset); free(h_centroids);
        return EXIT_FAILURE;
    }

    // Leitura em bloco super rápida de HPC
    size_t elementos_lidos = fread(h_dataset, sizeof(double), num_samples * num_features, file);
    fclose(file);

    if (elementos_lidos != (size_t)(num_samples * num_features)) {
        fprintf(stderr, "Erro: Arquivo binário corrompido ou incompleto.\n");
        free(h_dataset); free(h_centroids);
        return EXIT_FAILURE;
    }
    printf("Dataset carregado com sucesso! (%zu elementos inseridos na memória)\n\n", elementos_lidos);

    // Inicializa medição de performance
    double start_time = get_time_sec();

    // Transposição do dataset na CPU para Coalesced Access perfeito na GPU
    double *h_dataset_transposed = (double *)malloc(num_samples * num_features * sizeof(double));
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            h_dataset_transposed[j * num_samples + i] = h_dataset[i * num_features + j];
        }
    }

    // Inicialização dos Centroides (usa os primeiros K pontos do dataset)
    initialize_centroids(h_dataset, h_centroids, K, num_features);

    // Arrays auxiliares para atualização no Host
    double *h_new_centroids = (double *)malloc(K * num_features * sizeof(double));
    int *h_cluster_counts = (int *)malloc(K * sizeof(int));

    // Alocação de Memória no Device (GPU)
    double *d_dataset;
    double *d_new_centroids;
    int *d_cluster_counts;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dataset, num_samples * num_features * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_new_centroids, K * num_features * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cluster_counts, K * sizeof(int)));

    // Transfere o Dataset Transposto para o Device e inicializa a Memória Constante
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataset, h_dataset_transposed, num_samples * num_features * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_centroids_const, h_centroids, K * num_features * sizeof(double)));

    int iter = 0;
    int converged = 0;
    
    // Configuração de Blocos e Threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_samples + threadsPerBlock - 1) / threadsPerBlock;

    // Loop Principal do K-means
    while (iter < MAX_ITER && !converged) {
        
        // Zera acumuladores no Device antes da iteração
        CHECK_CUDA_ERROR(cudaMemset(d_new_centroids, 0, K * num_features * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemset(d_cluster_counts, 0, K * sizeof(int)));

        // Configura tamanho da Shared Memory Dinâmica (10 * 784 * 8 bytes para centroids + 10 * 4 bytes para counts)
        int shared_mem_size = (K * num_features * sizeof(double)) + (K * sizeof(int));
        cudaFuncSetAttribute(fused_kmeans_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

        // Kernel unificado de atribuição e acumulação
        fused_kmeans_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_dataset, d_new_centroids, d_cluster_counts, num_samples, num_features, K);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Traz as somas parciais de volta para o Host para finalizar a divisão
        CHECK_CUDA_ERROR(cudaMemcpy(h_new_centroids, d_new_centroids, K * num_features * sizeof(double), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(h_cluster_counts, d_cluster_counts, K * sizeof(int), cudaMemcpyDeviceToHost));

        // Passo D: Atualização e cálculo de convergência (no Host)
        double max_centroid_shift = 0.0;
        for (int k = 0; k < K; k++) {
            if (h_cluster_counts[k] > 0) {
                for (int j = 0; j < num_features; j++) {
                    double old_val = h_centroids[k * num_features + j];
                    double new_val = h_new_centroids[k * num_features + j] / h_cluster_counts[k];
                    h_centroids[k * num_features + j] = new_val;

                    double shift = fabs(old_val - new_val);
                    if (shift > max_centroid_shift) {
                        max_centroid_shift = shift;
                    }
                }
            }
        }

        // Envia os centroides atualizados de volta para a Memória Constante da GPU para a próxima iteração
        CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_centroids_const, h_centroids, K * num_features * sizeof(double)));

        iter++;
        if (max_centroid_shift < TOLERANCE) {
            converged = 1;
        }
    }

    double end_time = get_time_sec();
    printf("Processamento concluído!\n");
    printf("Iterações necessárias: %d\n", iter);
    printf("Tempo total de processamento do K-means: %.6f segundos\n", end_time - start_time);

    // Salvar centroides para visualização
    FILE *out_file = fopen("results/raw/centroids_cuda.bin", "wb");
    if (out_file) {
        fwrite(h_centroids, sizeof(double), K * num_features, out_file);
        fclose(out_file);
    }

    // Liberação de Memória
    free(h_dataset); free(h_dataset_transposed); free(h_centroids); free(h_new_centroids); free(h_cluster_counts);
    cudaFree(d_dataset); cudaFree(d_new_centroids); cudaFree(d_cluster_counts);

    return EXIT_SUCCESS;
}
