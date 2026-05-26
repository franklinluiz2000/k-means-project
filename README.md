# k-means-project

# Implementação e Análise de Desempenho do Algoritmo K-means Paralelo

Este repositório contém o projeto prático desenvolvido para a 3ª unidade da disciplina - Computação de Alto Desempenho, ministrada pelos professores Prof. Samuel Xavier de Souza e Prof. Carla Santana.

O objetivo do projeto é explorar diferentes paradigmas de programação paralela e avaliar o impacto do uso de arquiteturas de CPU e GPU no desempenho do algoritmo de agrupamento K-means.

## Equipe e Divisão de Tarefas
* Franklin Luiz da Cruz: Implementação Sequencial (Baseline) e Análise Teórica.
* Daniel Vitor de Oliveira Bezerra: Estudo, Arquitetura e Implementação Híbrida CPU (MPI + OpenMP).
* Raimundo Marciano de Freitas Neto: Estudo e Arquitetura de Paralelismo em GPU usando OpenMP.
* Luiz Gonzaga Gomes Neto: Estudo e Arquitetura de Programação Nativa em GPU usando CUDA.
* Luiz Gustavo de Souza Rego: Engenharia de Dados, Automação de Testes e Scripts no Cluster.

## Base de Dados (Dataset)
Titanic Dataset

## Versões Desenvolvidas
Para fins de comparação e análise de desempenho, o projeto contempla quatro abordagens distintas:
1. Sequencial: Código de referência base (baseline) para validação matemática.
2. Paralela com MPI + OpenMP: Abordagem com paralelismo híbrido para CPU (memória compartilhada e distribuída).
3. Paralela com OpenMP em GPU: Uso de diretivas de offloading para acelerar o processamento em GPU.
4. Paralela com CUDA: Implementação nativa de baixo nível voltada para GPUs NVIDIA.

## Métricas Analisadas
O desempenho de cada versão será mensurado no ambiente de supercomputação do cluster NPAD com base nas seguintes métricas de HPC:
* Tempo de Execução
* Speedup
* Eficiência
* Escalabilidade Forte e Fraca
* Avaliação de overheads de comunicação e tráfego de dados entre CPU e GPU