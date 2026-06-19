"""
Tabelas 3 e 4 do TCC — lê de kaggle_results/errors/

Tabela 3: MSE, RMSE e MAE por modelo (inferência direta, sem TTA)
Tabela 4: Wilcoxon Signed-Rank (Original vs demais)

Uso:
    python analytics/wilcoxon_test.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
import json
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent
STAT_DIR = ROOT_DIR / "kaggle_results" / "errors"

MODEL_ORDER = ['original', 'tcc_I', 'tcc_II', 'resnet', 'mobilenet_v2']
MODEL_LABEL = {
    'original':     'Original (CNN-TC)',
    'tcc_I':        'Diff1',
    'tcc_II':       'Diff2',
    'resnet':       'ResNet50',
    'mobilenet_v2': 'MobileNetV2',
}
COMPARISONS = ['tcc_I', 'tcc_II', 'resnet', 'mobilenet_v2']


def load_csv(key):
    path = STAT_DIR / key / "errors.csv"
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Tabela 3
# ---------------------------------------------------------------------------

def print_tabela3():
    print()
    print("=" * 62)
    print("  TABELA 3 - Comparacao dos modelos de predicao")
    print("=" * 62)
    print(f"  {'Modelo':<26} {'RMSE':>7} {'MAE':>7} {'MSE':>10}")
    print("  " + "-" * 58)
    for key in MODEL_ORDER:
        try:
            df = load_csv(key)
        except FileNotFoundError:
            print(f"  {MODEL_LABEL[key]:<26}  N/A (arquivo ausente)")
            continue
        e    = df['Erro_Bruto'].values
        mse  = float(np.mean(e ** 2))
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(e)))
        print(f"  {MODEL_LABEL[key]:<26} {rmse:>7.4f} {mae:>7.4f} {mse:>10.4f}")
    print("=" * 62)
    print("  RMSE e MAE em nos (inferencia direta, sem TTA)")


# ---------------------------------------------------------------------------
# Tabela 4
# ---------------------------------------------------------------------------

def run_wilcoxon(eo, em):
    valid = ~(np.isnan(eo) | np.isnan(em))
    eo, em = eo[valid], em[valid]
    stat, p = wilcoxon(eo, em, alternative='two-sided')
    pct     = 100.0 * (em < eo).sum() / len(eo)
    return float(stat), float(p), float(pct), int(len(eo))


def print_tabela4():
    print()
    print("=" * 80)
    print("  TABELA 4 - Wilcoxon Signed-Rank (Original vs demais)")
    print("=" * 80)

    try:
        df_orig = load_csv('original')
    except FileNotFoundError as e:
        print(f"  ERRO: {e}")
        return []

    ae_orig = df_orig['Erro_Absoluto'].values
    orig_label = MODEL_LABEL['original']
    print(f"  {orig_label:<22} Media={np.mean(ae_orig):.4f}  "
          f"Mediana={np.median(ae_orig):.4f}  Desvio={np.std(ae_orig):.4f}  "
          f"N={len(ae_orig)}")
    print()
    print(f"  {'Modelo':<22} {'Media':>8} {'Mediana':>8} {'Desvio':>8} "
          f"{'% Melhor':>10} {'p-value':>12} {'Sig?':>5}")
    print("  " + "-" * 76)

    all_results = []
    for key in COMPARISONS:
        try:
            df_other = load_csv(key)
        except FileNotFoundError:
            print(f"  {MODEL_LABEL[key]:<22}  N/A (arquivo ausente)")
            continue

        merged = df_orig.merge(
            df_other,
            on=['ID_Ciclone', 'Data_Hora'],
            suffixes=('_orig', '_other'),
        )
        if len(merged) < len(df_orig):
            print(f"  AVISO: alinhamento parcial para {key}: "
                  f"{len(merged)}/{len(df_orig)} amostras")

        eo = merged['Erro_Absoluto_orig'].values
        em = merged['Erro_Absoluto_other'].values
        stat, p, pct, n = run_wilcoxon(eo, em)

        mean_m   = float(np.mean(em[~np.isnan(em)]))
        median_m = float(np.median(em[~np.isnan(em)]))
        std_m    = float(np.std(em[~np.isnan(em)]))
        sig      = 'Sim*' if p < 0.05 else 'Nao'

        print(f"  {MODEL_LABEL[key]:<22} {mean_m:>8.4f} {median_m:>8.4f} "
              f"{std_m:>8.4f} {pct:>9.1f}% {p:>12.6f} {sig:>5}")

        all_results.append({
            'model':        MODEL_LABEL[key],
            'n':            n,
            'mean_orig':    float(np.mean(eo[~np.isnan(eo)])),
            'mean_other':   mean_m,
            'median_orig':  float(np.median(eo[~np.isnan(eo)])),
            'median_other': median_m,
            'std_orig':     float(np.std(eo[~np.isnan(eo)])),
            'std_other':    std_m,
            'pct_better':   pct,
            'statistic':    stat,
            'p_value':      p,
            'significant':  p < 0.05,
            'timestamp':    datetime.now().isoformat(),
        })

    print("=" * 80)
    print("  * Significante ao nivel alpha = 0,05")
    return all_results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print_tabela3()
    results = print_tabela4()

    if results:
        out = STAT_DIR.parent / "wilcoxon_summary.json"
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  JSON salvo em: {out}")


if __name__ == "__main__":
    main()
