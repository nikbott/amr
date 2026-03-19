import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    # Caminhos padrão esperados caso não sejam passados via argumento
    openmp_csv = '../openmp/openmp_results.csv'
    cuda_csv = '../cuda/cuda_results.csv'
    
    # Se os caminhos foram passados via argumentos de linha de comando
    if len(sys.argv) >= 3:
        openmp_csv = sys.argv[1]
        cuda_csv = sys.argv[2]
        
    has_openmp = os.path.exists(openmp_csv)
    has_cuda = os.path.exists(cuda_csv)
    
    if not has_openmp and not has_cuda:
        print(f"Erro: Nenhum dos arquivos de resultados encontrados.")
        print(f"Esperados: {openmp_csv} e {cuda_csv}")
        return

    # Preparar figura com subplots Lado a Lado (1 linha, 2 colunas)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- GRÁFICO 1: OPENMP ---
    if has_openmp:
        print(f"Lendo dados OpenMP de {openmp_csv}...")
        df_omp = pd.read_csv(openmp_csv)
        
        # Tirar a média das repetições
        mean_omp = df_omp.groupby('Threads')['Time_ms'].mean().reset_index()
        mean_omp = mean_omp.sort_values('Threads')
        
        # Converter para segundos para ficar igual a referência da imagem
        mean_omp['Time_s'] = mean_omp['Time_ms'] / 1000.0
        
        ax1 = axes[0]
        ax1.plot(mean_omp['Threads'], mean_omp['Time_s'], marker='o', linestyle='-', color='black', label='Nível 20')
        ax1.set_title('Tempo de Execução vs. Número de Threads (OpenMP)')
        ax1.set_xlabel('Número de Threads')
        ax1.set_ylabel('Tempo de Execução (s)')
        
        # Escala logarítmica (base 2) nos eixos como na imagem
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log', base=2)
        ax1.grid(True, linestyle='-', alpha=0.7)
        ax1.legend()
        
        # Ticks formatados
        ax1.set_xticks(mean_omp['Threads'])
        ax1.set_xticklabels([f'$2^{{{int(np.log2(x))}}}$' for x in mean_omp['Threads']])
        
    else:
        print(f"Aviso: {openmp_csv} não encontrado. Pulando gráfico do OpenMP.")
        axes[0].axis('off')

    # --- GRÁFICO 2: CUDA ---
    if has_cuda:
        print(f"Lendo dados CUDA de {cuda_csv}...")
        df_cuda = pd.read_csv(cuda_csv)
        
        # Tirar a média das repetições
        # Nota: O CSV do CUDA tem a coluna 'tpb' para Threads Por Bloco
        # Vamos assumir 'tpb' baseado no script de edição do colab.
        if 'tpb' in df_cuda.columns:
            mean_cuda = df_cuda.groupby('tpb')['Time_ms'].mean().reset_index()
            mean_cuda = mean_cuda.sort_values('tpb')
            
            # Converter para segundos
            mean_cuda['Time_s'] = mean_cuda['Time_ms'] / 1000.0
            
            ax2 = axes[1]
            ax2.plot(mean_cuda['tpb'], mean_cuda['Time_s'], marker='o', linestyle='-', color='black', label='Nível 20')
            ax2.set_title('Tempo de Execução vs. Número de Threads por Bloco (CUDA)')
            ax2.set_xlabel('Número de Threads por Bloco')
            ax2.set_ylabel('Tempo de Execução (s)')
            
            # Escala logarítmica (base 2) nos eixos como na imagem
            ax2.set_xscale('log', base=2)
            ax2.set_yscale('log', base=2)
            ax2.grid(True, linestyle='-', alpha=0.7)
            ax2.legend()
            
            # Ticks formatados
            ax2.set_xticks(mean_cuda['tpb'])
            ax2.set_xticklabels([f'$2^{{{int(np.log2(x))}}}$' for x in mean_cuda['tpb']])
        else:
            print("Aviso: Formato do CSV do CUDA inesperado.")
            axes[1].axis('off')
    else:
        print(f"Aviso: {cuda_csv} não encontrado. Pulando gráfico do CUDA. (Não se esqueça de realizar o download do Colab e salvá-lo nesta pasta!)")
        axes[1].axis('off')
        
    # Adicionando (a) e (b) abaixo dos gráficos como na imagem
    if has_openmp: axes[0].text(0.5, -0.15, '(a)', transform=axes[0].transAxes, fontsize=12, ha='center')
    if has_cuda: axes[1].text(0.5, -0.15, '(b)', transform=axes[1].transAxes, fontsize=12, ha='center')
    
    plt.tight_layout()
    
    # Salvar o gráfico final
    output_png = '../results/performance_comparison_graph.png'
    # Cria a pasta caso não exista
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nGráfico salvo com sucesso em: {output_png}")
    # plt.show() # Descomente para ver a janela
    
if __name__ == '__main__':
    main()
