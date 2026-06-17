import numpy as np
import matplotlib.pyplot as plt
import os

def load_centroids(filepath, K=10, num_features=784):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.float64)
    if len(data) != K * num_features:
        print(f"Erro: Tamanho inesperado no arquivo {filepath} (len = {len(data)})")
        return None
    return data.reshape((K, int(np.sqrt(num_features)), int(np.sqrt(num_features))))

def main():
    # Caminhos baseados na raiz do projeto (onde o script deve ser executado)
    seq_file = 'results/raw/centroids_sequential.bin'
    cuda_file = 'results/raw/centroids_cuda.bin'
    mpi_file = 'results/raw/centroids_mpi.bin'

    seq_centroids = load_centroids(seq_file)
    cuda_centroids = load_centroids(cuda_file)
    mpi_centroids = load_centroids(mpi_file)

    if seq_centroids is None or cuda_centroids is None or mpi_centroids is None:
        print("Arquivos de centroides não encontrados. Execute o test_correctness.sh no cluster primeiro.")
        return

    # Compara numericamente para garantir a precisão
    diff_cuda = np.abs(seq_centroids - cuda_centroids).max()
    diff_mpi = np.abs(seq_centroids - mpi_centroids).max()
    print(f"Diferença máxima absoluta (Seq vs CUDA): {diff_cuda:.10e}")
    print(f"Diferença máxima absoluta (Seq vs MPI): {diff_mpi:.10e}")
    
    if diff_cuda < 1e-6 and diff_mpi < 1e-6:
        print("-> SUCESSO: Os centroides são matematicamente idênticos nas 3 versões!")
    else:
        print("-> AVISO: Existe uma diferença matemática nos resultados.")

    # Gera a imagem comparativa (agora com 3 linhas)
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    
    for k in range(10):
        # Sequencial na linha superior
        axes[0, k].imshow(seq_centroids[k], cmap='gray')
        axes[0, k].axis('off')
        if k == 0:
            axes[0, k].set_title('Sequencial', fontsize=12, pad=20, rotation=0, loc='center')
        
        # MPI no meio
        axes[1, k].imshow(mpi_centroids[k], cmap='gray')
        axes[1, k].axis('off')
        if k == 0:
            axes[1, k].set_title('MPI+OpenMP', fontsize=12, pad=20, rotation=0, loc='center')

        # CUDA na linha inferior
        axes[2, k].imshow(cuda_centroids[k], cmap='gray')
        axes[2, k].axis('off')
        if k == 0:
            axes[2, k].set_title('CUDA', fontsize=12, pad=20, rotation=0, loc='center')
    
    plt.suptitle('Comparação Visual dos Centroides (K=10)', fontsize=16)
    plt.tight_layout()
    
    os.makedirs('results/figures', exist_ok=True)
    out_path = 'results/figures/centroids_comparison.png'
    plt.savefig(out_path, dpi=150)
    print(f"Imagem visual comparativa salva em {out_path}")

if __name__ == '__main__':
    main()
