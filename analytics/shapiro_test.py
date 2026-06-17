import os
from pathlib import Path
import pandas as pd
import scipy.stats as stats
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Inicializar console do Rich para formatação visual premium
console = Console()

def run_shapiro_test():
    root_dir = Path(__file__).parent.parent
    
    # 1. Definir a lista de caminhos dos arquivos errors.csv de cada modelo
    csv_paths = [
        root_dir / "result" / "statistical_test" / "original" / "errors.csv",
        root_dir / "result" / "statistical_test" / "tcc_I" / "errors.csv",
        root_dir / "result" / "statistical_test" / "tcc_II" / "errors.csv",
        root_dir / "result" / "statistical_test" / "resnet" / "errors.csv",
        root_dir / "result" / "statistical_test" / "mobilenet_v2" / "errors.csv",
    ]
    
    console.print(Panel.fit(
        "[bold cyan]TESTE DE NORMALIDADE DE SHAPIRO-WILK MULTI-MODELOS[/bold cyan]\n"
        "[dim]Objetivo: Avaliar se os erros de predição dos modelos seguem uma distribuição normal.[/dim]",
        border_style="cyan"
    ))
    
    # Tabela de Resultados final
    results_table = Table(title="Resumo dos Resultados - Teste de Shapiro-Wilk", border_style="blue")
    results_table.add_column("Modelo", style="cyan", justify="left")
    results_table.add_column("Métrica", style="magenta", justify="left")
    results_table.add_column("N (Amostras)", justify="right")
    results_table.add_column("Estatística W", justify="right")
    results_table.add_column("p-value", justify="right")
    results_table.add_column("Distribuição Normal? (α = 0.05)", justify="center")
    
    analyzed_count = 0
    
    # Iterar sobre cada arquivo e rodar os testes
    for csv_file in csv_paths:
        if not csv_file.exists():
            # Silenciosamente pular ou avisar se não existir
            console.print(f"[yellow]⚠️ Arquivo de erros não encontrado (pulando): {csv_file.relative_to(root_dir)}[/yellow]")
            continue
            
        model_name = csv_file.parent.name
        
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            console.print(f"[red]❌ Erro ao ler {csv_file.name} para o modelo {model_name}: {e}[/red]")
            continue
            
        analyzed_count += 1
        
        for col in ['Erro_Bruto', 'Erro_Absoluto']:
            if col not in df.columns:
                continue
                
            data = df[col].dropna().values
            
            if len(data) == 0:
                continue
                
            # Teste Shapiro-Wilk
            # Se N > 5000, o SciPy calcula mas emite um alerta de precisão.
            if len(data) > 5000:
                stat_w, p_val = stats.shapiro(data)
                obs_suffix = " (N > 5000)"
            else:
                stat_w, p_val = stats.shapiro(data)
                obs_suffix = ""
                
            p_val_str = f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.4f}"
            is_normal = "[green]Sim[/green]" if p_val > 0.05 else "[red]Não[/red]"
            
            results_table.add_row(
                model_name,
                col + obs_suffix,
                f"{len(data)}",
                f"{stat_w:.4f}",
                p_val_str,
                is_normal
            )
            
    console.print()
    
    if analyzed_count == 0:
        console.print("[red]❌ Nenhum arquivo 'errors.csv' da lista pôde ser carregado.[/red]")
        return
        
    # Exibir tabela com todos os resultados consolidados
    console.print(results_table)
    console.print()
    
    # Explicação e Conclusão Teórica
    teoria = (
        "[bold yellow]Interpretação Estatística:[/bold yellow]\n"
        "• [bold]Hipótese Nula (H0):[/bold] Os dados seguem uma distribuição normal.\n"
        "• [bold]Hipótese Alternativa (H1):[/bold] Os dados NÃO seguem uma distribuição normal.\n"
        "• Se [bold]p-value > 0.05[/bold]: Não rejeitamos H0 (os erros são normais).\n"
        "• Se [bold]p-value < 0.05[/bold]: Rejeitamos H0 (os erros NÃO são normais).\n\n"
        "[bold cyan]Nota Acadêmica para o TCC:[/bold cyan]\n"
        "1. [bold]Erro Absoluto:[/bold] Por definição, erros absolutos ($|Erro|$) são truncados em zero (assimétricos), "
        "o que faz com que o Shapiro-Wilk sempre aponte a [red]Não[/red] normalidade.\n"
        "2. [bold]Erro Bruto (Resíduos):[/bold] Em deep learning em larga escala, resíduos dificilmente seguem uma distribuição normal perfeita. "
        "A constatação de não-normalidade geral ([red]Não[/red]) nos resíduos de todos os modelos fundamenta a necessidade metodológica "
        "de usar testes de hipótese não-paramétricos (como o [bold]Teste de Wilcoxon[/bold]) para compará-los."
    )
    
    console.print(Panel(teoria, title="[bold green]Guia de Análise[/bold green]", border_style="green"))

if __name__ == "__main__":
    run_shapiro_test()
