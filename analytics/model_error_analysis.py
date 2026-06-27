"""
Distribuição dos erros absolutos por modelo.

Lê os errors.csv gerados pelo notebook Kaggle (com Rotation Blending / TTA)
e plota a distribuição de frequência dos erros absolutos de todos os modelos.

Uso:
    python analytics/model_error_analysis.py

Saída:
    kaggle_results/plots/error_dist_{modelo}.png  — histograma individual
    kaggle_results/plots/error_dist_all.png       — grade com todos os modelos
    kaggle_results/error_stats.json               — estatísticas por modelo
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR  = Path(__file__).parent.parent
ERRORS_DIR = ROOT_DIR / "kaggle_results" / "errors"
PLOTS_DIR  = ROOT_DIR / "kaggle_results" / "plots"

MODEL_ORDER = ['original', 'tcc_I', 'tcc_II', 'resnet', 'mobilenet_v2']
MODEL_LABEL = {
    'original':     'Original (CNN-TC)',
    'tcc_I':        'CNN-TC-Δ1',
    'tcc_II':       'CNN-TC-Δ2',
    'resnet':       'ResNet50',
    'mobilenet_v2': 'MobileNetV2',
}



def load_errors(key):
    path = ERRORS_DIR / key / "errors.csv"
    if not path.exists():
        raise FileNotFoundError(f"errors.csv não encontrado: {path}")
    df = pd.read_csv(path)
    return df['Erro_Absoluto'].dropna().values


def plot_single(ax, errors, label, color='steelblue'):
    """Histograma de erros absolutos num eixo matplotlib."""
    mean_val = np.mean(errors)
    median_val = np.median(errors)
    std_val = np.std(errors)

    ax.hist(errors, bins=50, edgecolor='black', linewidth=0.4,
            alpha=0.75, color=color)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.8,
               label=f'Média: {mean_val:.2f}')
    ax.axvline(median_val, color='orange', linestyle=':', linewidth=1.8,
               label=f'Mediana: {median_val:.2f}')
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xlabel('Erro Absoluto (nós)', fontsize=9)
    ax.set_ylabel('Frequência', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    stats = (
        f"N={len(errors)}\n"
        f"Média:  {mean_val:.4f}\n"
        f"Mediana:{median_val:.4f}\n"
        f"Desvio: {std_val:.4f}\n"
        f"Mín:    {errors.min():.4f}\n"
        f"Máx:    {errors.max():.4f}"
    )
    ax.text(0.97, 0.97, stats, transform=ax.transAxes,
            va='top', ha='right', fontsize=7.5, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    colors = ['steelblue', 'seagreen', 'darkorange', 'mediumpurple', 'crimson']
    loaded = []
    all_stats = {}

    # ── Plots individuais ──
    for key in MODEL_ORDER:
        try:
            errors = load_errors(key)
        except FileNotFoundError as e:
            print(f"  ⚠️  {e} — pulando.")
            continue

        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        color = colors[MODEL_ORDER.index(key) % len(colors)]
        plot_single(ax, errors, MODEL_LABEL[key], color)
        plt.tight_layout()
        out = PLOTS_DIR / f"error_dist_{key}.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Salvo: {out}")
        loaded.append((key, errors, color))
        all_stats[key] = {
            'label':   MODEL_LABEL[key],
            'n':       int(len(errors)),
            'mean':    float(np.mean(errors)),
            'median':  float(np.median(errors)),
            'std':     float(np.std(errors)),
            'min':     float(errors.min()),
            'max':     float(errors.max()),
            'p25':     float(np.percentile(errors, 25)),
            'p75':     float(np.percentile(errors, 75)),
            'p90':     float(np.percentile(errors, 90)),
            'p95':     float(np.percentile(errors, 95)),
        }

    if not loaded:
        print("Nenhum errors.csv encontrado em kaggle_results/errors/")
        return

    # ── Grade com todos os modelos ──
    if len(loaded) > 1:
        n = len(loaded)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), dpi=100)
        axes_flat = np.array(axes).flatten()
        for i, (key, errors, color) in enumerate(loaded):
            plot_single(axes_flat[i], errors, MODEL_LABEL[key], color)
        for j in range(len(loaded), len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle('Distribuição dos Erros Absolutos — Todos os Modelos\n(Predições com Rotation Blending / TTA)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out_all = PLOTS_DIR / "error_dist_all.png"
        plt.savefig(out_all, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Salvo: {out_all}")

    # ── Salvar estatísticas em JSON ──
    out_json = ROOT_DIR / "kaggle_results" / "error_stats.json"
    with open(out_json, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"  Salvo: {out_json}")

    # ── Tabela resumo no terminal ──
    print()
    print("=" * 62)
    print("  DISTRIBUIÇÃO DOS ERROS ABSOLUTOS (com TTA)")
    print("=" * 62)
    print(f"  {'Modelo':<26} {'N':>5} {'Média':>7} {'Mediana':>8} {'Desvio':>8}")
    print("  " + "-" * 58)
    for key, errors, _ in loaded:
        print(f"  {MODEL_LABEL[key]:<26} {len(errors):>5} "
              f"{np.mean(errors):>7.4f} {np.median(errors):>8.4f} {np.std(errors):>8.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
