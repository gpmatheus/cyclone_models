"""
Plota curvas de treinamento (loss e val_loss) dos 5 modelos do TCC.

Gera 2 figuras:
  result/plots/training_loss.png      — loss do conjunto de treino
  result/plots/validation_loss.png    — loss do conjunto de validação

Uso:
    python plot/plot_training.py
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR  = Path(__file__).parent.parent
PLOTS_DIR = ROOT_DIR / "result" / "plots"

MODELS = {
    'original':     ('result/original/history.pkl',                    'Original (CNN-TC)'),
    'tcc_I':        ('result/tcc_I/history.pkl',                       'CNN-TC-Δ1'),
    'tcc_II':       ('result/tcc_II/2/history.pkl',                    'CNN-TC-Δ2'),
    'resnet':       ('result/resnet/tcc_ii_preprocess/history.pkl',    'ResNet50'),
    'mobilenet_v2': ('result/mobilenetv2/history.pkl',                 'MobileNetV2'),
}

COLORS = {
    'original':     '#1f77b4',
    'tcc_I':        '#2ca02c',
    'tcc_II':       '#ff7f0e',
    'resnet':       '#9467bd',
    'mobilenet_v2': '#d62728',
}


def load_history(rel_path):
    path = ROOT_DIR / rel_path
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_plot(metric_key, ylabel, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    for key, (pkl_path, label) in MODELS.items():
        try:
            h = load_history(pkl_path)
        except FileNotFoundError:
            print(f"  ⚠️  {pkl_path} não encontrado — pulando.")
            continue

        values = h[metric_key]
        epochs = np.arange(1, len(values) + 1)
        ax.plot(epochs, values, color=COLORS[key], linewidth=1.8,
                label=f'{label}  (ep={len(values)})')

    ax.set_xlabel('Época', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Salvo: {out_path}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    make_plot(
        metric_key='loss',
        ylabel='Loss (MSE)',
        title='Loss — Conjunto de Treino',
        out_path=PLOTS_DIR / 'training_loss.png',
    )

    make_plot(
        metric_key='val_loss',
        ylabel='Loss (MSE)',
        title='Loss — Conjunto de Validação',
        out_path=PLOTS_DIR / 'validation_loss.png',
    )


if __name__ == '__main__':
    main()
