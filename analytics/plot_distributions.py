import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuração de Estilo Visual (Design Aesthetics)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.facecolor'] = '#f8fafc'  # Cinza bem claro (Slate 50)
plt.rcParams['axes.facecolor'] = '#ffffff'    # Branco para o fundo do plot
plt.rcParams['axes.edgecolor'] = '#cbd5e1'    # Borda suave (Slate 300)
plt.rcParams['grid.color'] = '#f1f5f9'        # Grid suave (Slate 100)
plt.rcParams['text.color'] = '#1e293b'        # Texto escuro (Slate 800)
plt.rcParams['axes.labelcolor'] = '#334155'   # Labels de eixos (Slate 700)
plt.rcParams['xtick.color'] = '#64748b'       # Ticks (Slate 500)
plt.rcParams['ytick.color'] = '#64748b'

# Paleta de Cores Harmônica e Moderna
COLORS = {
    'train': '#3b82f6',  # Azul Moderno (Slate Blue)
    'valid': '#f57c00',  # Laranja Quente (Amber)
    'test': '#10b981'    # Verde Esmeralda (Emerald)
}
LABELS = {
    'train': 'Treinamento (2003–2014)',
    'valid': 'Validação (2015–2016)',
    'test': 'Teste (2017)'
}

def get_year(time_val) -> int:
    """Extrai o ano do campo time de forma robusta."""
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

def main():
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data"
    plot_dir = root_dir / "result" / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("  ANÁLISE DE DISTRIBUIÇÃO DAS VELOCIDADES DOS CICLONES (Vmax)")
    print("=" * 70)
    
    # 1. Carregar tabelas de metadados ('info') dos 3 arquivos HDF5
    raw_files = [
        "TCIR-ATLN_EPAC_WPAC.h5",
        "TCIR-ALL_2017.h5",
        "TCIR-CPAC_IO_SH.h5"
    ]
    
    info_frames = []
    
    print("\n[1/4] Carregando tabelas de metadados...")
    for filename in raw_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  ⚠ Arquivo não encontrado: {filename} (pulando)")
            continue
        
        print(f"  Lendo chave 'info' do arquivo {filename}...")
        try:
            df = pd.read_hdf(str(filepath), key="info", mode="r")
            print(f"    -> Carregadas {len(df)} linhas.")
            info_frames.append(df)
        except Exception as e:
            print(f"  ❌ Erro ao ler {filename}: {e}")
            
    if not info_frames:
        print("❌ Nenhum dado foi carregado. Verifique os arquivos na pasta data/.")
        return
        
    # Concatenar todos os DataFrames
    combined_info = pd.concat(info_frames, ignore_index=True)
    print(f"  Total de registros combinados: {len(combined_info)}")
    
    # 2. Divisão dos dados usando o critério temporal
    print("\n[2/4] Dividindo dados por ano (critério de data.py)...")
    years = combined_info["time"].apply(get_year)
    
    train_mask = (years >= 2003) & (years <= 2014)
    valid_mask = (years >= 2015) & (years <= 2016)
    test_mask = (years == 2017)
    
    # Extrair velocidades (Vmax)
    train_vmax = combined_info.loc[train_mask, "Vmax"].dropna().to_numpy()
    valid_vmax = combined_info.loc[valid_mask, "Vmax"].dropna().to_numpy()
    test_vmax = combined_info.loc[test_mask, "Vmax"].dropna().to_numpy()
    
    # Exibir estatísticas descritivas
    datasets = {
        'train': train_vmax,
        'valid': valid_vmax,
        'test': test_vmax
    }
    
    print("\nEstatísticas Descritivas (Vmax em knots):")
    for key, data in datasets.items():
        if len(data) == 0:
            print(f"  {LABELS[key]}: Sem dados")
            continue
        print(f"  {LABELS[key]}:")
        print(f"    Amostras: {len(data)}")
        print(f"    Média:    {np.mean(data):.2f} kt")
        print(f"    Desvio P: {np.std(data):.2f} kt")
        print(f"    Mínimo:   {np.min(data):.1f} kt")
        print(f"    Mediana:  {np.median(data):.1f} kt")
        print(f"    Máximo:   {np.max(data):.1f} kt")
        
    # 3. Geração de Plots
    print("\n[3/4] Gerando gráficos...")
    
    # Plot 1: KDE + Histograma sobrepostos
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    fig.patch.set_facecolor('#f8fafc')
    
    for key, data in datasets.items():
        if len(data) == 0:
            continue
        # Histograma suave com KDE
        sns.histplot(
            data, 
            kde=True, 
            stat="density", 
            bins=30, 
            color=COLORS[key], 
            label=LABELS[key],
            alpha=0.25, 
            edgecolor=COLORS[key],
            linewidth=1.2,
            line_kws={"linewidth": 2.5},
            ax=ax
        )
        
    ax.set_title("Distribuição da Velocidade Máxima do Vento (Vmax) por Split", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Velocidade (Vmax) [knots]", fontsize=12, labelpad=10)
    ax.set_ylabel("Densidade de Probabilidade", fontsize=12, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", framealpha=0.9, fontsize=10)
    
    # Remover bordas desnecessárias (Despine)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    plot_path1 = plot_dir / "cyclone_velocity_kde_histogram.png"
    plt.savefig(plot_path1, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Salvo: {plot_path1.name}")
    
    # Plot 2: Boxplot Comparativo
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor('#f8fafc')
    
    plot_data = [datasets['train'], datasets['valid'], datasets['test']]
    plot_labels = [LABELS['train'], LABELS['valid'], LABELS['test']]
    
    bp = ax.boxplot(
        plot_data, 
        patch_artist=True, 
        notch=True,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#334155", "markersize": 6},
        flierprops={"marker": "o", "markerfacecolor": "#94a3b8", "markeredgecolor": "none", "markersize": 3, "alpha": 0.4},
        boxprops={"linewidth": 1.5},
        whiskerprops={"color": "#64748b", "linewidth": 1.2},
        capprops={"color": "#64748b", "linewidth": 1.2},
        medianprops={"color": "#1e293b", "linewidth": 2}
    )
    
    # Colorir as caixas de acordo com a paleta
    for patch, color in zip(bp['boxes'], [COLORS['train'], COLORS['valid'], COLORS['test']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor(color)
        
    ax.set_xticklabels(plot_labels, fontsize=11, fontweight="medium")
    ax.set_title("Resumo Estatístico da Velocidade (Vmax) por Split", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Velocidade (Vmax) [knots]", fontsize=12, labelpad=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    plot_path2 = plot_dir / "cyclone_velocity_boxplot.png"
    plt.savefig(plot_path2, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Salvo: {plot_path2.name}")
    
    # Plot 3: Violin Plot
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor('#f8fafc')
    
    # Preparar DataFrame para seaborn
    violin_df_list = []
    for key in ['train', 'valid', 'test']:
        temp_df = pd.DataFrame({
            'Vmax': datasets[key],
            'Split': LABELS[key]
        })
        violin_df_list.append(temp_df)
    violin_df = pd.concat(violin_df_list, ignore_index=True)
    
    sns.violinplot(
        data=violin_df,
        x='Split',
        y='Vmax',
        hue='Split',
        palette=[COLORS['train'], COLORS['valid'], COLORS['test']],
        density_norm='width',
        inner='quartile',
        legend=False,
        ax=ax
    )
    
    ax.set_title("Violin Plot da Velocidade (Vmax) por Split", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Split", fontsize=12, labelpad=10)
    ax.set_ylabel("Velocidade (Vmax) [knots]", fontsize=12, labelpad=10)
    ax.set_xticklabels([LABELS['train'], LABELS['valid'], LABELS['test']], fontsize=11, fontweight="medium")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    plot_path3 = plot_dir / "cyclone_velocity_violin.png"
    plt.savefig(plot_path3, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"  ✓ Salvo: {plot_path3.name}")
    
    print("\n[4/4] Processo concluído com sucesso!")
    print("=" * 70)

if __name__ == "__main__":
    main()
