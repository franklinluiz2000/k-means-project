#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <cuda_runtime.h>

#define MAX_ITER 300
#define TOLERANCE 1e-4

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

// Kernel 1: Atribuição de pontos aos clusters (Cálculo de Distância)
__global__ void assign_clusters_kernel(const double *dataset, const double *centroids, int *assignments,
                                       int num_samples, int num_features, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_samples) {
        double min_dist = DBL_MAX;
        int closest_cluster = 0;

        for (int k = 0; k < K; k++) {
            double dist = 0.0;
            for (int j = 0; j < num_features; j++) {
                double diff = dataset[i * num_features + j] - centroids[k * num_features + j];
                dist += diff * diff;
            }

            if (dist < min_dist) {
                min_dist = dist;
                closest_cluster = k;
            }
        }
        assignments[i] = closest_cluster;
    }
}

// Kernel 2: Acumulação para novos centroides usando atomicAdd
__global__ void accumulate_centroids_kernel(const double *dataset, const int *assignments, 
                                            double *new_centroids, int *cluster_counts,
                                            int num_samples, int num_features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_samples) {
        int cluster_id = assignments[i];
        
        // Atualiza a contagem do cluster
        atomicAdd(&cluster_counts[cluster_id], 1);
        
        // Atualiza as dimensões somando os pontos
        for (int j = 0; j < num_features; j++) {
            atomicAdd(&new_centroids[cluster_id * num_features + j], dataset[i * num_features + j]);
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

    // Inicialização dos Centroides (usa os primeiros K pontos do dataset)
    initialize_centroids(h_dataset, h_centroids, K, num_features);

    // Arrays auxiliares para atualização no Host
    double *h_new_centroids = (double *)malloc(K * num_features * sizeof(double));
    int *h_cluster_counts = (int *)malloc(K * sizeof(int));

    // Alocação de Memória no Device (GPU)
    double *d_dataset;
    double *d_centroids;
    double *d_new_centroids;
    int *d_assignments;
    int *d_cluster_counts;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_dataset, num_samples * num_features * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_centroids, K * num_features * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_new_centroids, K * num_features * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_assignments, num_samples * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_cluster_counts, K * sizeof(int)));

    // Transfere o Dataset e Centroides Iniciais do Host para o Device
    CHECK_CUDA_ERROR(cudaMemcpy(d_dataset, h_dataset, num_samples * num_features * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, K * num_features * sizeof(double), cudaMemcpyHostToDevice));

    int iter = 0;
    int converged = 0;
    
    // Configuração de Blocos e Threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_samples + threadsPerBlock - 1) / threadsPerBlock;

    // Loop Principal do K-means
    while (iter < MAX_ITER && !converged) {
        
        // Passo A: Kernel de atribuição
        assign_clusters_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_dataset, d_centroids, d_assignments, num_samples, num_features, K);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Espera o kernel finalizar para garantir integridade

        // Passo B e C: Zera acumuladores no Device
        CHECK_CUDA_ERROR(cudaMemset(d_new_centroids, 0, K * num_features * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMemset(d_cluster_counts, 0, K * sizeof(int)));

        // Kernel de acumulação (soma posições e contagem)
        accumulate_centroids_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_dataset, d_assignments, d_new_centroids, d_cluster_counts, num_samples, num_features);
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

        // Envia os centroides atualizados de volta para a GPU para a próxima iteração
        CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, K * num_features * sizeof(double), cudaMemcpyHostToDevice));

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
    free(h_dataset); free(h_centroids); free(h_new_centroids); free(h_cluster_counts);
    cudaFree(d_dataset); cudaFree(d_centroids); cudaFree(d_new_centroids); cudaFree(d_assignments); cudaFree(d_cluster_counts);

    return EXIT_SUCCESS;
}
