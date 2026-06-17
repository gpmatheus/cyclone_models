"""
Teste de Wilcoxon Signed-Rank para Comparação de Erros de Modelos
Compara os erros absolutos de predição entre dois modelos usando teste não-paramétrico.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
import json
from datetime import datetime

def load_error_data(filepath):
    """Carrega dados de erros do arquivo CSV."""
    df = pd.read_csv(filepath)
    print(f"  ✓ Carregado: {filepath}")
    print(f"    Amostras: {len(df)}")
    return df

def perform_wilcoxon_test(errors1, errors2, model1_name, model2_name):
    """Realiza o Teste de Wilcoxon Signed-Rank."""
    print(f"\n{'='*70}")
    print(f"  TESTE DE WILCOXON SIGNED-RANK")
    print(f"  Modelo 1: {model1_name}")
    print(f"  Modelo 2: {model2_name}")
    print(f"{'='*70}")
    
    # Remover NaN se houver
    valid_mask = ~(np.isnan(errors1) | np.isnan(errors2))
    errors1_clean = errors1[valid_mask]
    errors2_clean = errors2[valid_mask]
    
    print(f"\nAmostras válidas: {len(errors1_clean)}")
    
    # Estatísticas descritivas
    print(f"\nEstatísticas Descritivas:")
    print(f"  {model1_name}:")
    print(f"    Média:         {np.mean(errors1_clean):.4f}")
    print(f"    Mediana:       {np.median(errors1_clean):.4f}")
    print(f"    Desvio Padrão: {np.std(errors1_clean):.4f}")
    print(f"    Mín/Máx:       {np.min(errors1_clean):.4f} / {np.max(errors1_clean):.4f}")
    
    print(f"\n  {model2_name}:")
    print(f"    Média:         {np.mean(errors2_clean):.4f}")
    print(f"    Mediana:       {np.median(errors2_clean):.4f}")
    print(f"    Desvio Padrão: {np.std(errors2_clean):.4f}")
    print(f"    Mín/Máx:       {np.min(errors2_clean):.4f} / {np.max(errors2_clean):.4f}")
    
    # Calcular diferenças
    differences = errors1_clean - errors2_clean
    
    print(f"\nDiferenças (Modelo 1 - Modelo 2):")
    print(f"    Média:         {np.mean(differences):.4f}")
    print(f"    Mediana:       {np.median(differences):.4f}")
    
    # Realizar teste de Wilcoxon
    statistic, p_value = wilcoxon(errors1_clean, errors2_clean, alternative='two-sided')
    
    print(f"\nResultados do Teste de Wilcoxon Signed-Rank:")
    print(f"    Estatística W: {statistic:.4f}")
    print(f"    p-value:       {p_value:.6f}")
    print(f"    Nível α:       0.05")
    
    # Interpretação
    if p_value < 0.05:
        print(f"\n  ✓ Resultado: SIGNIFICATIVO")
        print(f"    Os erros entre os modelos são significativamente diferentes (p < 0.05)")
        
        # Determinar qual é menor
        if np.median(errors1_clean) < np.median(errors2_clean):
            print(f"    → {model1_name} tem erros MENORES que {model2_name}")
        else:
            print(f"    → {model2_name} tem erros MENORES que {model1_name}")
    else:
        print(f"\n  ✗ Resultado: NÃO SIGNIFICATIVO")
        print(f"    Os erros entre os modelos NÃO são significativamente diferentes (p ≥ 0.05)")
    
    # Contar amostras com melhora
    melhora_modelo1 = (errors1_clean < errors2_clean).sum()
    melhora_modelo2 = (errors2_clean < errors1_clean).sum()
    empate = (errors1_clean == errors2_clean).sum()
    
    print(f"\nComparação Amostra por Amostra:")
    print(f"    {model1_name} melhor: {melhora_modelo1} ({100*melhora_modelo1/len(errors1_clean):.1f}%)")
    print(f"    {model2_name} melhor: {melhora_modelo2} ({100*melhora_modelo2/len(errors1_clean):.1f}%)")
    print(f"    Empate:           {empate} ({100*empate/len(errors1_clean):.1f}%)")
    
    return {
        "modelo1": model1_name,
        "modelo2": model2_name,
        "amostras": len(errors1_clean),
        "wilcoxon_statistic": float(statistic),
        "p_value": float(p_value),
        "significante": bool(p_value < 0.05),
        "media_modelo1": float(np.mean(errors1_clean)),
        "media_modelo2": float(np.mean(errors2_clean)),
        "mediana_modelo1": float(np.median(errors1_clean)),
        "mediana_modelo2": float(np.median(errors2_clean)),
        "std_modelo1": float(np.std(errors1_clean)),
        "std_modelo2": float(np.std(errors2_clean)),
        "melhora_modelo1": int(melhora_modelo1),
        "melhora_modelo2": int(melhora_modelo2),
        "empate": int(empate),
        "timestamp": datetime.now().isoformat()
    }

def create_summary_table(results):
    """Cria e exibe tabela formatada em texto com resumo dos resultados."""
    print("\n" + "="*80)
    print("RESUMO COMPARATIVO - TESTE DE WILCOXON SIGNED-RANK".center(80))
    print("="*80)
    
    # Cabeçalho
    print(f"{'Métrica':<30} {results['modelo1']:<20} {results['modelo2']:<20}")
    print("-"*80)
    
    # Dados estatísticos
    print(f"{'Média de Erros':<30} {results['media_modelo1']:>19.4f} {results['media_modelo2']:>19.4f}")
    print(f"{'Mediana de Erros':<30} {results['mediana_modelo1']:>19.4f} {results['mediana_modelo2']:>19.4f}")
    print(f"{'Desvio Padrão':<30} {results['std_modelo1']:>19.4f} {results['std_modelo2']:>19.4f}")
    print("-"*80)
    
    # Comparação amostra por amostra
    pct_modelo1 = 100 * results['melhora_modelo1'] / results['amostras']
    pct_modelo2 = 100 * results['melhora_modelo2'] / results['amostras']
    pct_empate = 100 * results['empate'] / results['amostras']
    
    print(f"{'Melhor em (amostras)':<30} {results['melhora_modelo1']:>19} {results['melhora_modelo2']:>19}")
    print(f"{'Percentual (%)':<30} {pct_modelo1:>19.1f}% {pct_modelo2:>19.1f}%")
    print(f"{'Empates':<30} {results['empate']:>39}")
    print("-"*80)
    
    # Resultados do teste
    print(f"{'Estatística W (Wilcoxon)':<30} {results['wilcoxon_statistic']:>39.4f}")
    print(f"{'p-value':<30} {results['p_value']:>39.6f}")
    print(f"{'Nível de Significância (α)':<30} {'0.05 (5%)':>39}")
    print(f"{'Resultado':<30} {'SIGNIFICATIVO ✓' if results['significante'] else 'NÃO SIGNIFICATIVO ✗':>39}")
    print("="*80 + "\n")

def save_results(results, output_path):
    """Salva resultados em arquivo JSON."""
    output_file = Path(output_path) / "wilcoxon_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Resultados salvos em {output_file}")

def main():
    """Execução principal."""
    root_dir = Path(__file__).parent.parent
    
    # Caminhos dos arquivos
    file1_path = root_dir / "result" / "statistical_test" / "original" / "errors.csv"
    # file2_path = root_dir / "result" / "statistical_test" / "tcc_I" / "errors.csv"
    # file2_path = root_dir / "result" / "statistical_test" / "tcc_II" / "errors.csv"
    # file2_path = root_dir / "result" / "statistical_test" / "resnet" / "errors.csv"
    file2_path = root_dir / "result" / "statistical_test" / "mobilenet_v2" / "errors.csv"
    output_dir = root_dir / "result" / "plot"
    
    # Verificar se os arquivos existem
    if not file1_path.exists():
        print(f"❌ Arquivo não encontrado: {file1_path}")
        return
    if not file2_path.exists():
        print(f"❌ Arquivo não encontrado: {file2_path}")
        return
    
    print("="*70)
    print("  COMPARAÇÃO DE ERROS DE MODELOS - TESTE DE WILCOXON SIGNED-RANK")
    print("="*70)
    
    # Carregar dados
    print("\n[1/4] Carregando dados de erros...")
    df1 = load_error_data(str(file1_path))
    df2 = load_error_data(str(file2_path))
    
    # Ordenar ambos os DataFrames para garantir alinhamento perfeito por Ciclone e Data/Hora
    print("\n[2/4] Ordenando e garantindo o alinhamento dos dados...")
    if 'ID_Ciclone' in df1.columns and 'Data_Hora' in df1.columns:
        df1 = df1.sort_values(by=['ID_Ciclone', 'Data_Hora']).reset_index(drop=True)
    if 'ID_Ciclone' in df2.columns and 'Data_Hora' in df2.columns:
        df2 = df2.sort_values(by=['ID_Ciclone', 'Data_Hora']).reset_index(drop=True)
        
    print(f"  Amostras em {file1_path.parent.name}: {len(df1)}")
    print(f"  Amostras em {file2_path.parent.name}: {len(df2)}")
    
    # Verificar alinhamento final
    if 'ID_Ciclone' in df1.columns and 'ID_Ciclone' in df2.columns and 'Data_Hora' in df1.columns and 'Data_Hora' in df2.columns:
        match_id = (df1['ID_Ciclone'] == df2['ID_Ciclone']).all()
        match_time = (df1['Data_Hora'] == df2['Data_Hora']).all()
        if match_id and match_time:
            print("  ✓ Sucesso: Todos os IDs de ciclone e Data_Hora estão perfeitamente pareados entre os dois arquivos.")
        else:
            print("  ⚠️  Atenção: Mesmo após ordenação, os arquivos ainda possuem dados divergentes (ex. quantidade de linhas ou chaves diferentes)!")
            mismatches = (df1['ID_Ciclone'] != df2['ID_Ciclone']).sum()
            print(f"  Quantidade de linhas com IDs divergentes: {mismatches}")
    else:
        print("  ⚠️  Aviso: Colunas de ID_Ciclone ou Data_Hora ausentes em um dos arquivos. Não foi possível verificar o pareamento físico.")
        
    errors1 = df1['Erro_Absoluto'].values
    errors2 = df2['Erro_Absoluto'].values
    
    # Realizar teste
    print("\n[3/4] Realizando Teste de Wilcoxon Signed-Rank...")
    results = perform_wilcoxon_test(errors1, errors2, "Original", "TCC-I")
    
    # Exibir tabela resumida
    print("\n[4/4] Exibindo resumo dos resultados...")
    create_summary_table(results)
    
    # Salvar resultados
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(results, str(output_dir))
    
    print(f"\n{'='*70}")
    print("✓ Análise concluída com sucesso!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
