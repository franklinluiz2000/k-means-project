# Guia de Reprodutibilidade: Benchmarks K-means HPC

Este documento descreve o passo a passo exato para recriar localmente ou em outro nó de acesso do supercomputador a pesquisa exaustiva que cruzou as arquiteturas Sequencial, GPU (CUDA) e Cluster Massivo (MPI+OpenMP).

---

## 1. Conectando-se ao Ambiente
Assumimos que você possui acesso ao cluster NPAD via SSH.
Conecte-se ao nó de login e navegue até a pasta do projeto:
```bash
ssh seu_usuario@sc2.npad.ufrn.br
cd k-means-project
```

---

## 2. Geração dos Datasets (Cargas Variáveis)
Para testar a escalabilidade, você precisa forjar os blocos binários contendo de 17.500 amostras até 560.000 amostras do Fashion MNIST. Na raiz do projeto, execute o pipeline de dados em Python:
```bash
# Caso o ambiente virtual ainda não exista, crie-o:
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib

# Baixe as imagens base e gere o binário root (70k amostras puras)
python scripts/data_engineering.py

# Multiplique os dados gerando as réplicas de estresse
python scripts/generate_datasets.py
```
Isso criará arquivos que vão desde `fashion_mnist_17.5k.bin` até monstruosos 3.3 GB do `fashion_mnist_560k.bin` na pasta `data/`.

---

## 3. Disparando a Bateria Massiva no Slurm
A pesquisa exaustiva exige o uso de partições físicas distantes dentro do Datacenter (A fila de GPU V100 e a fila Massiva de CPUs AMD). Submetemos as duas filas simultaneamente:

**A. Submissão do Job de GPU (CUDA + Sequencial Baseline):**
```bash
sbatch scripts/run_gpu_comprehensive.sh
```
> **Nota:** Como o Baseline Sequencial é testado em apenas 1 CPU e chega a calcular bases de meio milhão de registros, esse Job pode durar **cerca de 50 minutos** até sua finalização.

**B. Submissão do Job Clusterizado (MPI + OpenMP):**
```bash
sbatch scripts/run_mpi_comprehensive.sh
```
> **Nota:** Esse script engatará **64 processadores simultâneos divididos em 4 nós** (Afinidade de GPU e core resolvidas por `bind-to none`). Costuma fritar bases de 560k em pouco mais de 10 segundos.

---

## 4. Monitorando a Conclusão
Fique de olho na finalização dos jobs. Anote os IDs submetidos e verifique na fila de alocação do Slurm:
```bash
squeue -u $USER
```
Para ver o andamento em "tempo real" preenchendo os relatórios das planilhas:
```bash
tail -f results/csv/gpu_comprehensive.csv
tail -f results/csv/mpi_comprehensive.csv
```

---

## 5. Análise de Dados e Gráficos (Local)
Terminado os processamentos de HPC, não tente renderizar gráficos localmente no nó SSH. Baixe os resultados para a sua máquina pessoal:

No seu terminal Linux/Mac (MÁQUINA LOCAL):
```bash
# Trazendo os CSVs pra máquina
rsync -avz seu_usuario@sc2.npad.ufrn.br:/home/seu_usuario/k-means-project/results/csv/ ./results/csv/
```

Agora cruze os dados utilizando o script Python na raiz do repositório local:
```bash
source .venv/bin/activate
python scripts/analyze_results.py
```

Pronto! Os gráficos finais (incluindo `comprehensive_time.png` e `comprehensive_speedup.png`) bem como a tabela resumo serão salvos na pasta `/results/figures/` e `/results/csv/summary_table.csv`, provando a eficiência dos seus nós!

---

## 6. Corretude e Validação Visual (Centroides)
Para garantir que a aceleração matemática não compromete a corretude do algoritmo, todas as três versões (Sequencial, CUDA e MPI) exportam suas matrizes matemáticas finais de centroides em formato bruto binário na pasta `results/raw/` (como `centroids_sequential.bin`, `centroids_cuda.bin`, etc).

Para comparar as diferenças mínimas entre eles, desenhá-los em tela e certificar de que todos chegaram ao mesmíssimo cluster de peças de roupa (Fashion MNIST):

1. Traga a pasta `raw` da mesma forma como trouxe os CSVs:
```bash
rsync -avz seu_usuario@sc2.npad.ufrn.br:/home/seu_usuario/k-means-project/results/raw/ ./results/raw/
```

2. Rode o script de análise e renderização de corretude (Máquina Local):
```bash
python scripts/visualize_centroids.py
```
Isso validará o Delta numérico de erro (indicando a precisão idêntica entre os algoritmos de CPU e GPU) e plotará uma imagem consolidada de todos os centróides gerados.
