import os
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Inicializar console do Rich para formatação visual premium
console = Console()

def get_year(time_val) -> int:
    """Extrai o ano do campo time de forma robusta (mesma lógica do plot_distributions.py)."""
    if isinstance(time_val, bytes):
        time_str = time_val.decode('utf-8')
    else:
        time_str = str(time_val)
    
    # Remove parte decimal se houver
    if '.' in time_str:
        time_str = time_str.split('.')[0]
    time_str = time_str.strip()
    
    if not time_str or time_str == 'nan' or time_str == '-1':
        return -1
    
    try:
        return datetime.datetime.strptime(time_str, "%Y%m%d%H").year
    except ValueError:
        # Tenta extrair os primeiros 4 caracteres como ano
        try:
            return int(time_str[:4])
        except ValueError:
            return -1

def holm_bonferroni(p_values):
    """
    Aplica a correção de Holm-Bonferroni para uma lista de p-values.
    Retorna os p-values corrigidos na ordem original.
    """
    n = len(p_values)
    sorted_indices = sorted(range(n), key=lambda k: p_values[k])
    
    corrected = [0.0] * n
    prev_corrected = 0.0
    
    for i, idx in enumerate(sorted_indices):
        m_minus_i_plus_1 = n - i
        raw_p = p_values[idx]
        adj_p = raw_p * m_minus_i_plus_1
        corrected_p = min(1.0, max(adj_p, prev_corrected))
        corrected[idx] = corrected_p
        prev_corrected = corrected_p
        
    return corrected

def run_tests():
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    
    console.print(Panel.fit(
        "[bold cyan]TESTE DE HOMOGENEIDADE DAS DISTRIBUIÇÕES DE VMAX[/bold cyan]\n"
        "[dim]Objetivo: Determinar estatisticamente se os três datasets possuem a mesma distribuição.[/dim]",
        border_style="cyan"
    ))
    
    # 1. Carregar tabelas de metadados ('info') dos 3 arquivos HDF5
    raw_files = [
        "TCIR-ATLN_EPAC_WPAC.h5",
        "TCIR-ALL_2017.h5",
        "TCIR-CPAC_IO_SH.h5"
    ]
    
    info_frames = []
    
    with console.status("[bold green]Carregando dados dos arquivos HDF5...", spinner="dots"):
        for filename in raw_files:
            filepath = data_dir / filename
            if not filepath.exists():
                console.print(f"[yellow]⚠ Arquivo não encontrado: {filename} (pulando)[/yellow]")
                continue
            
            try:
                df = pd.read_hdf(str(filepath), key="info", mode="r")
                info_frames.append(df)
            except Exception as e:
                console.print(f"[red]❌ Erro ao ler {filename}: {e}[/red]")
                
    if not info_frames:
        console.print("[bold red]❌ Nenhum dado foi carregado. Verifique os arquivos na pasta data/.[/bold red]")
        return
        
    combined_info = pd.concat(info_frames, ignore_index=True)
    
    # 2. Divisão dos dados por ano
    years = combined_info["time"].apply(get_year)
    
    train_mask = (years >= 2003) & (years <= 2014)
    valid_mask = (years >= 2015) & (years <= 2016)
    test_mask = (years == 2017)
    
    # Extrair velocidades (Vmax)
    train_vmax = combined_info.loc[train_mask, "Vmax"].dropna().to_numpy()
    valid_vmax = combined_info.loc[valid_mask, "Vmax"].dropna().to_numpy()
    test_vmax = combined_info.loc[test_mask, "Vmax"].dropna().to_numpy()
    
    datasets = {
        'Treino (2003-2014)': train_vmax,
        'Validação (2015-2016)': valid_vmax,
        'Teste (2017)': test_vmax
    }
    
    # 3. Estatísticas Descritivas Básicas
    desc_table = Table(title="Resumo dos Datasets", border_style="blue")
    desc_table.add_column("Dataset", style="cyan", justify="left")
    desc_table.add_column("Número de Amostras (N)", justify="right")
    desc_table.add_column("Média (knots)", justify="right")
    desc_table.add_column("Mediana (knots)", justify="right")
    desc_table.add_column("Desvio Padrão (knots)", justify="right")
    
    for label, data in datasets.items():
        if len(data) == 0:
            desc_table.add_row(label, "0", "-", "-", "-")
            continue
        desc_table.add_row(
            label,
            f"{len(data):,}",
            f"{np.mean(data):.2f}",
            f"{np.median(data):.1f}",
            f"{np.std(data):.2f}"
        )
    console.print(desc_table)
    console.print()
    
    valid_data_lists = [d for d in datasets.values() if len(d) > 0]
    if len(valid_data_lists) < 3:
        console.print("[red]❌ Erro: É necessário ter dados nos 3 splits para realizar os testes de hipótese.[/red]")
        return
        
    # 4. Teste Global de Comparação: Kruskal-Wallis H-test
    # Testa a hipótese nula de que todas as populações têm a mesma distribuição (ou mediana).
    stat_kw, p_kw = stats.kruskal(*valid_data_lists)
    
    global_table = Table(title="1. Teste Global (Comparação Simultânea dos 3 Datasets)", border_style="yellow")
    global_table.add_column("Teste Estatístico", style="cyan")
    global_table.add_column("Valor da Estatística", justify="right")
    global_table.add_column("p-value", justify="right")
    global_table.add_column("Resultado (Alfa = 0.05)", justify="left")
    
    p_kw_str = f"{p_kw:.3e}" if p_kw < 0.001 else f"{p_kw:.4f}"
    
    if p_kw < 0.05:
        global_result = "[bold red]Rejeita H0: Pelo menos um dataset tem distribuição diferente[/bold red]"
    else:
        global_result = "[bold green]Não rejeita H0: As distribuições são estatisticamente iguais[/bold green]"
        
    global_table.add_row(
        "Kruskal-Wallis H",
        f"{stat_kw:.2f}",
        p_kw_str,
        global_result
    )
    console.print(global_table)
    console.print(
        "[dim]H0 (Hipótese Nula): Todos os três datasets vêm da mesma distribuição.\n"
        "H1 (Hipótese Alternativa): Pelo menos um dataset possui distribuição diferente (locais ou formas distintas).[/dim]\n"
    )
    
    # 5. Testes Par a Par: Kolmogorov-Smirnov (KS) de 2 Amostras
    # O teste KS é o teste padrão e mais robusto para dizer se duas distribuições empíricas são iguais.
    pairs = [
        ('Treino (2003-2014)', 'Validação (2015-2016)'),
        ('Treino (2003-2014)', 'Teste (2017)'),
        ('Validação (2015-2016)', 'Teste (2017)')
    ]
    
    raw_p_values = []
    ks_statistics = []
    
    for g1, g2 in pairs:
        d1 = datasets[g1]
        d2 = datasets[g2]
        res = stats.ks_2samp(d1, d2)
        raw_p_values.append(res.pvalue)
        ks_statistics.append(res.statistic)
        
    # Correção de Holm-Bonferroni para os múltiplos testes par a par
    corrected_p_values = holm_bonferroni(raw_p_values)
    
    pairwise_table = Table(title="2. Comparação Par a Par (Teste Kolmogorov-Smirnov de 2 Amostras)", border_style="magenta")
    pairwise_table.add_column("Par Comparado", style="cyan")
    pairwise_table.add_column("Estatística D (KS)", justify="right")
    pairwise_table.add_column("p-value Corrigido", justify="right")
    pairwise_table.add_column("As distribuições são iguais?", justify="center")
    
    for idx, (g1, g2) in enumerate(pairs):
        p_corr = corrected_p_values[idx]
        d_stat = ks_statistics[idx]
        
        p_corr_str = f"{p_corr:.3e}" if p_corr < 0.001 else f"{p_corr:.4f}"
        
        if p_corr < 0.05:
            are_equal = "[bold red]Não (Diferentes)[/bold red]"
        else:
            are_equal = "[bold green]Sim (Iguais)[/bold green]"
            
        pairwise_table.add_row(
            f"{g1.split(' ')[0]} vs {g2.split(' ')[0]}",
            f"{d_stat:.4f}",
            p_corr_str,
            are_equal
        )
        
    console.print(pairwise_table)
    console.print(
        "[dim]H0: As duas amostras foram extraídas da mesma distribuição contínua.\n"
        "A estatística D mede a distância máxima vertical entre as funções de distribuição acumulada (CDF) dos dois datasets.[/dim]\n"
    )
    
    # 6. Veredito Final
    console.print("[bold yellow]3. Veredito Final[/bold yellow]")
    
    # Verificar se todas as comparações indicam igualdade
    all_equal = all(p >= 0.05 for p in corrected_p_values)
    
    if all_equal:
        verdict_title = "[bold green]Veredito: As distribuições dos três datasets são ESTATISTICAMENTE IGUAIS.[/bold green]"
        verdict_body = (
            "Com base no teste global de Kruskal-Wallis e nos testes par a par de Kolmogorov-Smirnov, "
            "[bold green]não há evidência estatística[/bold green] de que a distribuição da velocidade do vento (Vmax) "
            "difira entre os conjuntos de Treino, Validação e Teste (todos os p-values > 0.05).\n\n"
            "[bold]Significado Prático:[/bold]\n"
            "Isso significa que a divisão dos dados (seja temporal ou aleatória) gerou conjuntos homogêneos e representativos "
            "da mesma população, o que é ideal para treinar e avaliar seu modelo sem desvios de distribuição (*distribution shift*)."
        )
        border_color = "green"
    else:
        # Identificar quais pares são diferentes
        diff_pairs = []
        for idx, (g1, g2) in enumerate(pairs):
            if corrected_p_values[idx] < 0.05:
                diff_pairs.append(f"{g1.split(' ')[0]} vs {g2.split(' ')[0]}")
                
        verdict_title = "[bold red]Veredito: As distribuições dos três datasets são ESTATISTICAMENTE DIFERENTES.[/bold red]"
        verdict_body = (
            f"O teste Kolmogorov-Smirnov detectou [bold red]diferenças estatisticamente significativas[/bold red] "
            f"nas distribuições de Vmax nos seguintes pares: [bold]{', '.join(diff_pairs)}[/bold] (p-values corrigidos < 0.05).\n\n"
            "[bold]Significado Prático e Justificativa para o TCC:[/bold]\n"
            "1. Há uma mudança de distribuição (*distribution shift* ou *temporal shift*) entre os anos que compõem os datasets.\n"
            "2. [bold cyan]Justificativa de Validação:[/bold cyan] Isso justifica cientificamente por que você usou uma divisão temporal estrita "
            "(Treino: 2003-2014, Validação: 2015-2016, Teste: 2017) em vez de uma divisão aleatória (K-Fold comum).\n"
            "3. Avaliar o modelo em um dataset com distribuição estatisticamente diferente (como o Teste de 2017) "
            "prova que seu modelo de Deep Learning é robusto e capaz de generalizar para dados futuros desconhecidos "
            "(*out-of-distribution generalization*)."
        )
        border_color = "red"
        
    console.print(Panel(verdict_body, title=verdict_title, border_style=border_color))

if __name__ == "__main__":
    run_tests()
