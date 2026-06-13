"""
Gera datasets de tamanhos variados para testes de escalabilidade fraca.

Replica o dataset Fashion MNIST original para criar versoes maiores,
e cria subsets menores para testes rapidos.

Uso: python scripts/generate_datasets.py
"""
import os
import sys
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
BASE_FILE = os.path.join(DATA_DIR, 'fashion_mnist_pure.bin')

NUM_FEATURES = 784
BASE_SAMPLES = 70000

DATASETS = {
    'fashion_mnist_7k.bin': 7000,
    'fashion_mnist_14k.bin': 14000,
    'fashion_mnist_35k.bin': 35000,
    'fashion_mnist_140k.bin': 140000,
    'fashion_mnist_280k.bin': 280000,
    'fashion_mnist_560k.bin': 560000,
}


def load_base_dataset() -> np.ndarray:
    if not os.path.exists(BASE_FILE):
        print(f"Erro: Dataset base '{BASE_FILE}' nao encontrado.")
        print("Execute 'python scripts/data_engineering.py' primeiro.")
        sys.exit(1)

    data = np.fromfile(BASE_FILE, dtype=np.float64)
    expected = BASE_SAMPLES * NUM_FEATURES
    if data.size != expected:
        print(f"Erro: Tamanho inesperado. Esperado {expected}, encontrado {data.size}")
        sys.exit(1)

    return data.reshape(BASE_SAMPLES, NUM_FEATURES)


def generate_dataset(base: np.ndarray, target_samples: int, output_path: str) -> None:
    if target_samples <= BASE_SAMPLES:
        subset = base[:target_samples]
    else:
        repeats = target_samples // BASE_SAMPLES
        remainder = target_samples % BASE_SAMPLES
        parts = [base] * repeats
        if remainder > 0:
            parts.append(base[:remainder])
        subset = np.concatenate(parts, axis=0)

    subset.tofile(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  {os.path.basename(output_path)}: {target_samples} amostras, {size_mb:.1f} MB")


def main() -> None:
    print("=== Geracao de Datasets para Escalabilidade ===\n")
    print(f"Carregando dataset base ({BASE_SAMPLES} amostras)...")
    base = load_base_dataset()
    print(f"Dataset base carregado: {base.shape}\n")

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Gerando datasets:")
    for filename, num_samples in sorted(DATASETS.items(), key=lambda x: x[1]):
        output_path = os.path.join(DATA_DIR, filename)
        generate_dataset(base, num_samples, output_path)

    print("\nTodos os datasets gerados com sucesso!")


if __name__ == '__main__':
    main()
