import numpy as np
import tensorflow as tf  # Ou use o 'keras' para obter o dataset

print("Baixando Fashion MNIST...")
(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()

# Junta os dados de treino (60k) e teste (10k) para ter o dataset completo (70.000 imagens)
dataset_completo = np.concatenate((x_train, x_test), axis=0)

# Cada imagem é 28x28 (784 pixels). Vamos achatar para um vetor de 784 posições
dataset_plano = dataset_completo.reshape(-1, 28 * 28)

# Normaliza os dados para float64 (double em C) entre 0.0 e 1.0 (padrão para algoritmos de distância)
dataset_normalizado = dataset_plano.astype(np.float64) / 255.0

print(importante := f"Formato final do dataset: {dataset_normalizado.shape}") # Deve ser (70000, 784)

# Salva tudo em um arquivo binário bruto
dataset_normalizado.tofile("fashion_mnist_pure.bin")
print("Arquivo 'fashion_mnist_pure.bin' gerado com sucesso!")