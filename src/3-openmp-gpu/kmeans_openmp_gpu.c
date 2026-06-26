#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <omp.h>

#define MAX_ITER 300
#define TOLERANCE 1e-4

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int num_samples = 70000;        // tamanho do fashion mnist
    int num_features = 784;         // quantidade de atributos do dataset
    int K = 10;                     
    char dataset_path[256] = "data/fashion_mnist_pure.bin"; 

    if (argc >= 2) {
        num_samples = atoi(argv[1]);
    }
    if (argc >= 3) {
        snprintf(dataset_path, sizeof(dataset_path), "%s", argv[2]);
    }

    printf("=== HPC K-means (OpenMP GPU Offloading) ===\n");
    printf("Configuração: %d amostras, %d dimensões, K=%d\n", num_samples, num_features, K);
    printf("Dataset: %s\n", dataset_path);

    double *dataset = (double *)malloc(num_samples * num_features * sizeof(double));
    double *centroids = (double *)malloc(K * num_features * sizeof(double));
    double *new_centroids = (double *)malloc(K * num_features * sizeof(double));
    int *cluster_counts = (int *)malloc(K * sizeof(int));

    // Leitura do binário com travas de erro obrigatórias
    FILE *f = fopen(dataset_path, "rb");
    if (f) {
        size_t elements_read = fread(dataset, sizeof(double), (size_t)num_samples * num_features, f);
        if (elements_read < (size_t)num_samples * num_features) {
            printf("Erro fatal: O arquivo binário leu menos elementos do que o esperado (%zu/%d).\n", 
                   elements_read, num_samples * num_features);
            fclose(f);
            free(dataset); free(centroids); free(new_centroids); free(cluster_counts);
            return 1;
        }
        fclose(f);
    } else {
        printf("Erro fatal: O arquivo '%s' não foi encontrado!\n", dataset_path);
        printf("Certifique-se de que o caminho está correto ou que o arquivo existe.\n");
        free(dataset); free(centroids); free(new_centroids); free(cluster_counts);
        return 1; 
    }

    // Inicialização dos centróides (usando as primeiras K amostras)
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < num_features; j++) {
            centroids[i * num_features + j] = dataset[i * num_features + j];
        }
    }

    double start_time = get_time_sec();

    int iter = 0;
    int converged = 0;

    // Aloca e mapeia os dados explicitamente na GPU
    // o dataset eh enviado, mas nao precisa retornar
    // o centroid inicializado vai, e precisa retornar
    // fases do calculado sao iniciadas diretamente na gpu e nao sao necessarias depois
    #pragma omp target data map(to: dataset[0:num_samples*num_features]) \
                            map(tofrom: centroids[0:K*num_features]) \
                            map(alloc: new_centroids[0:K*num_features], cluster_counts[0:K])
    {
        while (iter < MAX_ITER && !converged) {
            
            // Inicializa os acumuladores dentro da GPU em paralelo
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < K; i++) {
                cluster_counts[i] = 0;
                for (int j = 0; j < num_features; j++) {
                    new_centroids[i * num_features + j] = 0.0;
                }
            }

            // Loop principal: Mapeia as amostras nos cores da GPU
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < num_samples; i++) {
                double min_dist = DBL_MAX;
                int cluster_id = 0;

                for (int k = 0; k < K; k++) {
                    double dist = 0.0;
                    for (int j = 0; j < num_features; j++) {
                        double diff = dataset[i * num_features + j] - centroids[k * num_features + j];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        cluster_id = k;
                    }
                }

                // Proteção atômica contra condições de corrida na GPU
                #pragma omp atomic update
                cluster_counts[cluster_id]++;

                for (int j = 0; j < num_features; j++) {
                    #pragma omp atomic update
                    new_centroids[cluster_id * num_features + j] += dataset[i * num_features + j];
                }
            }

            // Copia os dados acumulados da GPU de volta para a CPU
            #pragma omp target update from(new_centroids[0:K*num_features], cluster_counts[0:K])

            // CPU calcula a média e valida a tolerância de convergência
            double max_centroid_shift = 0.0;
            for (int k = 0; k < K; k++) {
                if (cluster_counts[k] > 0) {
                    for (int j = 0; j < num_features; j++) {
                        double new_val = new_centroids[k * num_features + j] / cluster_counts[k];
                        double shift = fabs(centroids[k * num_features + j] - new_val);
                        if (shift > max_centroid_shift) max_centroid_shift = shift;
                        centroids[k * num_features + j] = new_val;
                    }
                }
            }

            if (max_centroid_shift < TOLERANCE) converged = 1;
            iter++;

            // Envia os centróides atualizados para a GPU começar a próxima iteração
            #pragma omp target update to(centroids[0:K*num_features])
        }
    } // Memória alocada na GPU é liberada aqui

    double end_time = get_time_sec();

    printf("Processamento concluído!\n");
    printf("Iterações necessárias: %d\n", iter);
    printf("Tempo total de processamento do K-means na GPU: %.6f segundos\n", end_time - start_time);

    // Salva o resultado final
    FILE *out_file = fopen("results/raw/centroids_openmp_gpu.bin", "wb");
    if (out_file) {
        fwrite(centroids, sizeof(double), K * num_features, out_file);
        fclose(out_file);
    }

    free(dataset);
    free(centroids);
    free(new_centroids);
    free(cluster_counts);
    return 0;
}