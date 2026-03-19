# Plano de Execução dos Experimentos AMR

Este documento descreve o plano de testes e execução para avaliar a escalabilidade térmica e de processamento das versões OpenMP e CUDA do projeto AMR.

## Parâmetros Base (Precisão Científica)

Para garantir precisão científica e volume de dados suficiente para observar os ganhos de paralelismo, fixaremos os seguintes níveis na estrutura Quadtree (como visto nos scripts anteriores):
- **MAX_LEVEL**: `20`
- **FINE_LEVEL**: `12`

## 1. Experimentos OpenMP (CPU)

O objetivo é medir a escalabilidade forte (Strong Scaling) alterando o número de threads enquanto o tamanho do problema permanece fixo.

**Configurações de Teste:**
- Threads: 2, 4, 8, 16, 32, 64, 128.
- Repetições: 5 vezes para cada configuração para calcular a média de tempo e remover anomalias.

**Execução:**
Um script automatizado (`openmp/run_experiments.sh`) foi criado para:
1. Compilar o código garantindo que a flag `-O3` e `-fopenmp` estejam ativadas.
2. Executar as 5 repetições para cada quantidade de threads.
3. Extrair o tempo total (em ms) de cada execução e salvar em um arquivo CSV (`results/openmp_benchmark.csv`).

## 2. Experimentos CUDA (GPU no Colab)

A execução CUDA será feita no Google Colab para aproveitar as GPUs gratuitas (ex: T4, P100 ou V100). O objetivo é avaliar a performance variando o tamanho e arranjo dos blocos.

**Configurações de Teste:**
- Threads por bloco (tpb): 32, 64, 128, 256, 512, 1024.
- Número de blocos: Será dinamicamente calculado para cada kernel com a fórmula padrão:
  `dim3 grid((dados_totais + tpb - 1) / tpb);`

**Passo-a-passo no Colab:**
1. Fazer upload da pasta `cuda/` (ou clonar o repositório).
2. Compilar o código usando `nvcc` com otimização: 
   `!nvcc -O3 -arch=sm_75 main.cu tree_kernels.cu -o amr_cuda` (Ajustar `-arch` caso a GPU não seja T4).
3. Executar o código alterando o parâmetro de threads por bloco. *Você precisará passar esse argumento pela linha de comando ou script no Colab.*
4. Salvar os resultados de tempo de execução para cada configuração de `tpb` em um arquivo ou copiá-los para um CSV manualmente.

## 3. Análise dos Dados e Speedup

Foi criado um script Python (`scripts/plot_speedup.py`) que:
1. Lê o CSV gerado.
2. Calcula o tempo médio das execuções para cada arranjo.
3. Calcula o **Speedup**, que é a razão entre o tempo da configuração base (menor número de threads) e o tempo das configurações mais paralelas.
4. Gera um gráfico salvando a imagem em `results/speedup_graph.png`.
