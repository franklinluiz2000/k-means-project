#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <omp.h>
#include <mpi.h>

#define MAX_ITER 300
#define TOLERANCE 1e-4

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int num_samples = 70000;
    int num_features = 784;
    int K = 10;

    // Validação para garantir fatias iguais
    if (num_samples % size != 0) {
        if (rank == 0) printf("Erro: 70000 deve ser divisível pelo número de processos MPI.\n");
        MPI_Finalize();
        return 1;
    }

    int local_samples = num_samples / size;

    double *dataset = NULL;
    double *centroids = (double *)malloc(K * num_features * sizeof(double));
    double *local_dataset = (double *)malloc(local_samples * num_features * sizeof(double));

    if (rank == 0) {
        printf("=== HPC K-means Híbrido (MPI + OpenMP) ===\n");
        printf("Configuração: %d amostras, %d dimensões, K=%d\n", num_samples, num_features, K);
        
        dataset = (double *)malloc(num_samples * num_features * sizeof(double));
        
        // Simulação da leitura do binário gerado pelo Python
        FILE *f = fopen("data/fashion_mnist_pure.bin", "rb");
        if (f) {
            fread(dataset, sizeof(double), num_samples * num_features, f);
            fclose(f);
        } else {
            printf("Aviso: data/fashion_mnist_pure.bin não encontrado. Preenchendo com dados aleatórios.\n");
            for (int i = 0; i < num_samples * num_features; i++) dataset[i] = (double)rand() / RAND_MAX;
        }

        // Inicialização dos centróides
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < num_features; j++) {
                centroids[i * num_features + j] = dataset[i * num_features + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime(); // Usamos o timer do MPI agora

    // O Processo 0 espalha as imagens
    MPI_Scatter(dataset, local_samples * num_features, MPI_DOUBLE,
                local_dataset, local_samples * num_features, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // O Processo 0 envia os centróides iniciais para todos
    MPI_Bcast(centroids, K * num_features, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int iter = 0;
    int converged = 0;

    while (iter < MAX_ITER && !converged) {
        double *local_new_centroids = (double *)calloc(K * num_features, sizeof(double));
        int *local_cluster_counts = (int *)calloc(K, sizeof(int));

        #pragma omp parallel
        {
            double *thread_new_centroids = (double *)calloc(K * num_features, sizeof(double));
            int *thread_cluster_counts = (int *)calloc(K, sizeof(int));

            #pragma omp for
            for (int i = 0; i < local_samples; i++) {
                double min_dist = DBL_MAX;
                int cluster_id = 0;

                for (int k = 0; k < K; k++) {
                    double dist = 0.0;
                    for (int j = 0; j < num_features; j++) {
                        double diff = local_dataset[i * num_features + j] - centroids[k * num_features + j];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        cluster_id = k;
                    }
                }

                thread_cluster_counts[cluster_id]++;
                for (int j = 0; j < num_features; j++) {
                    thread_new_centroids[cluster_id * num_features + j] += local_dataset[i * num_features + j];
                }
            }

            #pragma omp critical
            {
                for (int k = 0; k < K; k++) {
                    local_cluster_counts[k] += thread_cluster_counts[k];
                    for (int j = 0; j < num_features; j++) {
                        local_new_centroids[k * num_features + j] += thread_new_centroids[k * num_features + j];
                    }
                }
            }
            free(thread_new_centroids);
            free(thread_cluster_counts);
        }

        int *global_cluster_counts = (int *)calloc(K, sizeof(int));
        double *global_new_centroids = (double *)calloc(K * num_features, sizeof(double));

        MPI_Allreduce(local_cluster_counts, global_cluster_counts, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_new_centroids, global_new_centroids, K * num_features, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double max_centroid_shift = 0.0;
        for (int k = 0; k < K; k++) {
            if (global_cluster_counts[k] > 0) {
                for (int j = 0; j < num_features; j++) {
                    double new_val = global_new_centroids[k * num_features + j] / global_cluster_counts[k];
                    double shift = fabs(centroids[k * num_features + j] - new_val);
                    if (shift > max_centroid_shift) max_centroid_shift = shift;
                    centroids[k * num_features + j] = new_val;
                }
            }
        }

        if (max_centroid_shift < TOLERANCE) converged = 1;
        iter++;

        free(local_new_centroids); free(local_cluster_counts);
        free(global_new_centroids); free(global_cluster_counts);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Processamento Híbrido concluído em %d iterações.\n", iter);
        printf("Tempo total de processamento: %.6f segundos\n", end_time - start_time);
        free(dataset);
    }

    free(centroids);
    free(local_dataset);
    MPI_Finalize();
    return 0;
}