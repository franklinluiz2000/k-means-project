#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>

#define MAX_ITER 300
#define TOLERANCE 1e-4

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

    printf("=== HPC K-means Sequencial (Baseline) ===\n");
    printf("Dataset: %s\n", dataset_path);
    printf("Configuração: %d amostras, %d dimensões, K=%d\n\n", num_samples, num_features, K);

    // Alocação de Memória
    double *dataset = (double *)malloc(num_samples * num_features * sizeof(double));
    double *centroids = (double *)malloc(K * num_features * sizeof(double));
    int *assignments = (int *)malloc(num_samples * sizeof(int));

    if (!dataset || !centroids || !assignments) {
        fprintf(stderr, "Erro de alocação de memória.\n");
        return EXIT_FAILURE;
    }

    printf("Carregando dados do arquivo binário...\n");
    FILE *file = fopen(dataset_path, "rb");
    if (!file) {
        fprintf(stderr, "Erro: O arquivo '%s' não foi encontrado!\n", dataset_path);
        free(dataset); free(centroids); free(assignments);
        return EXIT_FAILURE;
    }

    // Leitura em bloco super rápida de HPC
    size_t elementos_lidos = fread(dataset, sizeof(double), num_samples * num_features, file);
    fclose(file);

    if (elementos_lidos != (size_t)(num_samples * num_features)) {
        fprintf(stderr, "Erro: Arquivo binário corrompido ou incompleto.\n");
        free(dataset); free(centroids); free(assignments);
        return EXIT_FAILURE;
    }
    printf("Dataset carregado com sucesso! (%zu elementos inseridos na memória)\n\n", elementos_lidos);

    // Inicializa medição de performance
    double start_time = get_time_sec();

    // Inicialização dos Centroides
    initialize_centroids(dataset, centroids, K, num_features);

    double *new_centroids = (double *)calloc(K * num_features, sizeof(double));
    int *cluster_counts = (int *)calloc(K, sizeof(int));

    int iter = 0;
    int converged = 0;

    // Loop Principal do K-means
    while (iter < MAX_ITER && !converged) {
        
        // Passo A: Atribuição de pontos aos clusters (Cálculo de Distância)
        for (int i = 0; i < num_samples; i++) {
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

        // Passo B e C: Reset e Acumulação para novos centroides
        for (int i = 0; i < K * num_features; i++) new_centroids[i] = 0.0;
        for (int i = 0; i < K; i++) cluster_counts[i] = 0;

        for (int i = 0; i < num_samples; i++) {
            int cluster_id = assignments[i];
            cluster_counts[cluster_id]++;
            for (int j = 0; j < num_features; j++) {
                new_centroids[cluster_id * num_features + j] += dataset[i * num_features + j];
            }
        }

        // Passo D: Atualização e cálculo de convergência
        double max_centroid_shift = 0.0;
        for (int k = 0; k < K; k++) {
            if (cluster_counts[k] > 0) {
                for (int j = 0; j < num_features; j++) {
                    double old_val = centroids[k * num_features + j];
                    double new_val = new_centroids[k * num_features + j] / cluster_counts[k];
                    centroids[k * num_features + j] = new_val;

                    double shift = fabs(old_val - new_val);
                    if (shift > max_centroid_shift) {
                        max_centroid_shift = shift;
                    }
                }
            }
        }

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
    FILE *out_file = fopen("results/raw/centroids_sequential.bin", "wb");
    if (out_file) {
        fwrite(centroids, sizeof(double), K * num_features, out_file);
        fclose(out_file);
    }

    // Liberação de Memória
    free(dataset); free(centroids); free(assignments);
    free(new_centroids); free(cluster_counts);

    return EXIT_SUCCESS;
}