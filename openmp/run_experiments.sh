#!/bin/bash

# Este script executa os testes experimentais da versão Sequencial e OpenMP do AMR
# fixando um valor alto de MAX_LEVEL para precisão científica e rodando com diferentes threads.
# Além disso, faz várias repetições para tirar a média de tempo.

# Configurações de precisão (Nível da malha)
MAX_LEVEL=20
FINE_LEVEL=12

# Array com número de threads a testar
THREADS=(2 4 8 16 32 64 128)

# Número de vezes que cada teste será executado (repetições)
REPETITIONS=5

# Arquivo de saída em formato CSV
RESULTS_CSV="openmp_results.csv"

# Prepara o arquivo CSV
echo "Threads,Repetition,Time_ms" > $RESULTS_CSV

echo "=== Iniciando Experimentos ==="
echo "MAX_LEVEL=$MAX_LEVEL, FINE_LEVEL=$FINE_LEVEL"
echo "Repetições por configuração: $REPETITIONS"
echo "Resultados serão salvos em $RESULTS_CSV"
echo "---------------------------------------"

# Compilar versão sequencial
echo "Compilando versão Sequencial..."
g++ -O3 -std=c++17 ../sequential/amr_seq.cpp -o ../sequential/amr_seq > /dev/null

# Executar versão sequencial
echo ">> Testando Versão Sequencial (Baseline com Threads=1)"
for (( i=1; i<=$REPETITIONS; i++ )); do
    echo -n "   Repetição $i/$REPETITIONS... "
    OUTPUT=$(../sequential/amr_seq --max_level $MAX_LEVEL --fine_level $FINE_LEVEL)
    TIME_MS=$(echo "$OUTPUT" | grep "Total execution time:" | awk '{print $4}')
    
    if [ -z "$TIME_MS" ]; then
        echo "ERRO: Não foi possível capturar o tempo."
        TIME_MS=-1
    else
        echo "${TIME_MS} ms"
    fi
    # Vamos salvar '1' thread como sendo a versão sequencial para servir de base no speedup
    echo "1,$i,$TIME_MS" >> $RESULTS_CSV
done
echo "---------------------------------------"


# Compilar o código OpenMP garantindo que está atualizado
echo "Compilando projeto OpenMP (Requer CMake/Make)..."
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. > /dev/null
make -j > /dev/null
cd ..

for t in "${THREADS[@]}"; do
    echo ">> Testando OpenMP com OMP_NUM_THREADS=$t"
    export OMP_NUM_THREADS=$t
    
    for (( i=1; i<=$REPETITIONS; i++ )); do
        echo -n "   Repetição $i/$REPETITIONS... "
        
        # Executa o programa e captura a saída
        OUTPUT=$(./build/amr --max_level $MAX_LEVEL --fine_level $FINE_LEVEL)
        
        # Extrai o tempo total da saída
        TIME_MS=$(echo "$OUTPUT" | grep "Total execution time:" | awk '{print $4}')
        
        if [ -z "$TIME_MS" ]; then
            echo "ERRO: Não foi possível capturar o tempo."
            TIME_MS=-1
        else
            echo "${TIME_MS} ms"
        fi
        
        # Salva no arquivo CSV
        echo "$t,$i,$TIME_MS" >> $RESULTS_CSV
    done
    echo "---------------------------------------"
done

echo "Experimentos concluídos com sucesso!"
echo "Resultados finais (CSV):"
cat $RESULTS_CSV
