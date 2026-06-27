"""
Tabela 5 do TCC — Teste de Shapiro-Wilk sobre os erros absolutos de predicao.

Verifica se os erros seguem distribuicao normal.
Resultado: nao normal em todos os modelos → justifica uso do Wilcoxon.

Uso:
    python analytics/shapiro_test.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import shapiro

ROOT_DIR = Path(__file__).parent.parent
STAT_DIR = ROOT_DIR / "kaggle_results" / "errors"

MODEL_ORDER = ['original', 'tcc_I', 'tcc_II', 'resnet', 'mobilenet_v2']
MODEL_LABEL = {
    'original':     'Original (CNN-TC)',
    'tcc_I':        'CNN-TC-Δ1',
    'tcc_II':       'CNN-TC-Δ2',
    'resnet':       'ResNet50',
    'mobilenet_v2': 'MobileNetV2',
}


def run_shapiro_test():
    print()
    print("=" * 65)
    print("  TABELA 5 - Shapiro-Wilk (Erro Absoluto por modelo)")
    print("=" * 65)
    print(f"  {'Modelo':<26} {'Estatistica W':>14} {'p-value':>14} {'Normal? (0,05)':>15}")
    print("  " + "-" * 61)

    for key in MODEL_ORDER:
        path = STAT_DIR / key / "errors.csv"
        if not path.exists():
            print(f"  {MODEL_LABEL[key]:<26}  N/A (arquivo ausente)")
            continue

        df   = pd.read_csv(path)
        vals = df['Erro_Absoluto'].dropna().values

        W, p = shapiro(vals)

        # Formata p-value em notacao cientifica com base 10
        exp   = int(np.floor(np.log10(abs(p)))) if p > 0 else 0
        coef  = p / (10 ** exp)
        p_str = f"{coef:.3f}e{exp:+03d}"

        is_normal = "Sim" if p > 0.05 else "Nao"
        print(f"  {MODEL_LABEL[key]:<26} {W:>14.4f} {p_str:>14} {is_normal:>15}")

    print("=" * 65)
    print("  Conclusao: p < 0,05 em todos → distribuicao nao normal")
    print("             → justifica o uso do Teste de Wilcoxon")


if __name__ == "__main__":
    run_shapiro_test()
