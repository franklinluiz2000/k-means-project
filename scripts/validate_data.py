"""
Valida a integridade dos arquivos binarios de dataset.

Verifica: tamanho do arquivo, range de valores, presenca de NaN/Inf.

Uso: python scripts/validate_data.py [arquivo.bin num_samples]
"""
import os
import sys
import glob
import numpy as np

NUM_FEATURES = 784


def validate_binary(filepath: str, expected_samples: int | None = None) -> bool:
    filename = os.path.basename(filepath)

    if not os.path.exists(filepath):
        print(f"  FAIL  {filename}: arquivo nao encontrado")
        return False

    file_size = os.path.getsize(filepath)
    total_doubles = file_size // 8

    if file_size % 8 != 0:
        print(f"  FAIL  {filename}: tamanho ({file_size} bytes) nao e multiplo de 8")
        return False

    if total_doubles % NUM_FEATURES != 0:
        print(f"  FAIL  {filename}: {total_doubles} doubles nao e divisivel por {NUM_FEATURES}")
        return False

    num_samples = total_doubles // NUM_FEATURES

    if expected_samples is not None and num_samples != expected_samples:
        print(f"  FAIL  {filename}: esperado {expected_samples} amostras, encontrado {num_samples}")
        return False

    data = np.fromfile(filepath, dtype=np.float64)

    if np.any(np.isnan(data)):
        print(f"  FAIL  {filename}: contem valores NaN")
        return False

    if np.any(np.isinf(data)):
        print(f"  FAIL  {filename}: contem valores Inf")
        return False

    min_val, max_val = data.min(), data.max()
    if min_val < -0.01 or max_val > 1.01:
        print(f"  FAIL  {filename}: valores fora do range [0,1] (min={min_val:.4f}, max={max_val:.4f})")
        return False

    size_mb = file_size / (1024 * 1024)
    print(f"  PASS  {filename}: {num_samples} amostras, {NUM_FEATURES} dims, "
          f"range [{min_val:.4f}, {max_val:.4f}], {size_mb:.1f} MB")
    return True


def main() -> None:
    print("=== Validacao de Datasets Binarios ===\n")

    if len(sys.argv) >= 3:
        filepath = sys.argv[1]
        expected = int(sys.argv[2])
        ok = validate_binary(filepath, expected)
        sys.exit(0 if ok else 1)

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    bin_files = sorted(glob.glob(os.path.join(data_dir, '*.bin')))

    if not bin_files:
        print("Nenhum arquivo .bin encontrado em data/")
        print("Execute 'python scripts/data_engineering.py' primeiro.")
        sys.exit(1)

    known_sizes = {
        'fashion_mnist_pure.bin': 70000,
        'fashion_mnist_7k.bin': 7000,
        'fashion_mnist_14k.bin': 14000,
        'fashion_mnist_35k.bin': 35000,
        'fashion_mnist_140k.bin': 140000,
        'fashion_mnist_280k.bin': 280000,
        'fashion_mnist_560k.bin': 560000,
    }

    passed = 0
    failed = 0
    for f in bin_files:
        basename = os.path.basename(f)
        expected = known_sizes.get(basename)
        if validate_binary(f, expected):
            passed += 1
        else:
            failed += 1

    print(f"\nResultado: {passed} PASS, {failed} FAIL")
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
