import os
import gzip
import numpy as np
import urllib.request

print("=== Engenharia de Dados: Baixando Fashion MNIST (Modo Leve HPC) ===")

# URLs oficiais dos arquivos compactados no GitHub do Fashion MNIST
base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
files = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz'
}

# Cria a pasta data se não existir
os.makedirs('data', exist_ok=True)

def load_mnist_images(filename):
    filepath = os.path.join('data', filename)
    if not os.path.exists(filepath):
        print(f"Baixando {filename}...")
        urllib.request.urlretrieve(base_url + filename, filepath)

    with gzip.open(filepath, 'rb') as f:
        # Os primeiros 16 bytes são metadados do arquivo IDX
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # Redimensiona para o número de imagens (cada uma tem 28x28 = 784 pixels)
    return data.reshape(-1, 28 * 28)

# Baixa e carrega os blocos de treino e teste
train_images = load_mnist_images(files['train_img'])
test_images = load_mnist_images(files['test_img'])

# Consolida o dataset completo (70.000 amostras, 784 features)
dataset_completo = np.concatenate((train_images, test_images), axis=0)

# Normaliza para float64 (double em C) entre 0.0 e 1.0
dataset_normalizado = dataset_completo.astype(np.float64) / 255.0

print(f"Formato final do dataset: {dataset_normalizado.shape}")

output_path = os.path.join('data', 'fashion_mnist_pure.bin')
dataset_normalizado.tofile(output_path)

print(f"Sucesso! Arquivo '{output_path}' gerado com sucesso!")