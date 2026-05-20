"""
SHAP Explainer com Rotation Blending (TTA) — Regressão de Vmax
================================================================
Aplica GradientExplainer do SHAP para mapas de atribuição pixel-wise
em um CNN de regressão que prediz a velocidade máxima (Vmax) de ciclones.

Estratégia TTA — idêntica ao gradcam.py:
  Para cada ângulo θ ∈ [0°, 360°) uniformes:
    1. Rotar imagem por θ
    2. Computar SHAP values na imagem rotacionada
    3. Contra-rotacionar os SHAP values por −θ (espaço original)
    4. Acumular e calcular média

Otimização de performance:
  - SHAP computado em BATCH por rotação (não imagem a imagem)
  - Reduz N_imgs × N_rots chamadas SHAP → N_batches × N_rots chamadas

Uso:
    python shap_explainer.py [--limit N] [--bg N] [--rotations N] [--batch N]

Variáveis de ambiente:
    MODEL_PATH, DATASET_PATH, OUTPUT_DIR
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import h5py
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
from scipy.stats import spearmanr
from typing import Tuple, Optional

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

keras = tf.keras

MODEL_PATH   = os.getenv("MODEL_PATH")   or "result/tcc_II/2/model.keras"
DATASET_PATH = os.getenv("DATASET_PATH") or "data/preprocessed/test.h5"
OUTPUT_DIR   = os.getenv("OUTPUT_DIR")   or "result/shap_output"
INPUT_SIZE   = 64


# ============================================================================
# Utilitários — idênticos ao gradcam.py / plot_result.py
# ============================================================================

def center_crop(images: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Crop centralizado (N, H, W, C) → (N, size, size, C)."""
    h, w = images.shape[1], images.shape[2]
    if h == size and w == size:
        return images
    sh, sw = (h - size) // 2, (w - size) // 2
    print(f"  Center-crop: {images.shape[1:3]} → ({size}, {size})")
    return images[:, sh:sh + size, sw:sw + size, :]


def rotate_image_tf(image: tf.Tensor, angle_deg: float, output_size: int) -> tf.Tensor:
    """Rotaciona tensor (H, W, C) usando transformação afim projetiva."""
    angle_rad = tf.cast(angle_deg * np.pi / 180.0, tf.float32)
    image_shape = tf.shape(image)[0:2]
    cx = tf.cast(image_shape[1] / 2, tf.float32)
    cy = tf.cast(image_shape[0] / 2, tf.float32)
    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)
    transform = tf.stack([
        cos_a, -sin_a, (1 - cos_a) * cx + sin_a * cy,
        sin_a,  cos_a, (1 - cos_a) * cy - sin_a * cx,
        0.0, 0.0,
    ])
    transform = tf.reshape(transform, [8])
    transform = tf.expand_dims(transform, 0)
    image_4d = tf.expand_dims(image, 0)
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image_4d, transforms=transform,
        output_shape=image_shape, interpolation="BILINEAR",
        fill_mode="REFLECT", fill_value=0.0,
    )
    rotated = tf.squeeze(rotated, 0)
    return tf.image.resize_with_crop_or_pad(rotated, output_size, output_size)


def normalize_display(img: np.ndarray) -> np.ndarray:
    """Normaliza imagem para [0,1] float."""
    img = img - img.min()
    mx = img.max()
    return (img / mx).astype(np.float32) if mx > 0 else img.astype(np.float32)


# ============================================================================
# Classe principal — SHAP com Rotation Blending
# ============================================================================

class SHAPRegressionTTA:
    """
    GradientExplainer com Rotation Blending para regressão de Vmax.

    Diferencial de performance: para cada ângulo de rotação, computa SHAP
    em batch (B imagens de uma vez), reduzindo o número total de chamadas
    SHAP de N × R para (N/batch_size) × R.
    """

    def __init__(self, model: keras.Model, background: np.ndarray, rotations: int = 10):
        self.model     = model
        self.rotations = rotations
        self.angles    = [i * 360.0 / rotations for i in range(rotations)]
        print(f"  Inicializando GradientExplainer  bg={len(background)}  rotações={rotations}")
        self.explainer = shap.GradientExplainer(model, background)

    # ------------------------------------------------------------------
    def _extract_shap(self, raw) -> np.ndarray:
        """Normaliza output do GradientExplainer para (N, H, W, C)."""
        if isinstance(raw, list):
            raw = raw[0]
        sv = np.array(raw, dtype=np.float32)
        # Regressão pode retornar (N, H, W, C, 1) — remover última dim
        if sv.ndim == 5 and sv.shape[-1] == 1:
            sv = sv[..., 0]
        return sv  # (N, H, W, C)

    # ------------------------------------------------------------------
    def _rotate_batch_numpy(self, images: np.ndarray, angle_deg: float, size: int) -> np.ndarray:
        """Rota batch de imagens numpy (B, H, W, C) pelo mesmo ângulo."""
        return np.array([
            rotate_image_tf(tf.cast(img, tf.float32), angle_deg, size).numpy()
            for img in images
        ], dtype=np.float32)

    def _counter_rotate_shap_batch(self, sv: np.ndarray, angle_deg: float, size: int) -> np.ndarray:
        """Contra-rotaciona SHAP values (B, H, W, C) por −angle_deg."""
        return np.array([
            rotate_image_tf(tf.cast(sv[i], tf.float32), -angle_deg, size).numpy()
            for i in range(len(sv))
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    def process_batch(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processa batch (B, H, W, C) com TTA completo.

        Returns:
            shap_avg : (B, H, W, C) float32 — SHAP médio sobre rotações
            pred_avg : (B,) float32 — predição média sobre rotações
        """
        B, H, W, C = images.shape
        shap_acc = np.zeros((B, H, W, C), dtype=np.float32)
        pred_acc = np.zeros(B, dtype=np.float32)

        for angle_deg in self.angles:
            # 1. Rotar todo o batch
            imgs_rot = self._rotate_batch_numpy(images, angle_deg, H)

            # 2. Predições e SHAP em batch
            preds = self.model.predict(imgs_rot, verbose=0)[:, 0]
            sv = self._extract_shap(self.explainer.shap_values(imgs_rot))  # (B,H,W,C)

            pred_acc += preds

            # 3. Contra-rotacionar SHAP values
            sv_back = self._counter_rotate_shap_batch(sv, angle_deg, H)
            shap_acc += sv_back

        return shap_acc / self.rotations, pred_acc / self.rotations

    # ------------------------------------------------------------------
    def process_all(
        self,
        images: np.ndarray,
        batch_size: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Processa todas as imagens em mini-batches com progresso."""
        N = len(images)
        n_batches = (N + batch_size - 1) // batch_size
        all_sv, all_pred = [], []

        print(f"\n  Calculando SHAP para {N} amostras | batch={batch_size} | R={self.rotations}")
        print(f"  Total de chamadas SHAP: {n_batches * self.rotations}")

        for b in range(n_batches):
            s = b * batch_size
            e = min(s + batch_size, N)
            sv, pred = self.process_batch(images[s:e])
            all_sv.append(sv)
            all_pred.append(pred)
            pct = (b + 1) / n_batches * 100
            print(f"  [{b+1:4d}/{n_batches}]  amostras {s:5d}–{e-1:5d}  {pct:5.1f}%", flush=True)

        return np.concatenate(all_sv), np.concatenate(all_pred)


# ============================================================================
# Visualizações
# ============================================================================

DARK_BG  = "#1a1a2e"
DARK_AX  = "#16213e"
WHITE    = "white"


def _dark_fig(*args, **kwargs):
    fig = plt.figure(*args, facecolor=DARK_BG, **kwargs)
    return fig


def _save(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Salvo: {path}")


# ------------------------------------------------------------------
def plot_global_mean(shap_maps: np.ndarray, output_dir: str) -> None:
    """Média global de |SHAP| por canal — revela padrões estruturais."""
    C = shap_maps.shape[-1]
    mean_abs = np.mean(np.abs(shap_maps), axis=0)  # (H, W, C)

    fig, axes = plt.subplots(1, C + 1, figsize=(5 * (C + 1), 4.5), facecolor=DARK_BG)
    maps    = [mean_abs[..., c] for c in range(C)] + [mean_abs.sum(-1)]
    titles  = [f"Canal {c}" for c in range(C)] + ["Soma canais"]

    for ax, title, m in zip(axes, titles, maps):
        ax.set_facecolor(DARK_AX)
        im = ax.imshow(m, cmap="hot", interpolation="bilinear")
        ax.set_title(title, color=WHITE, fontsize=10)
        ax.axis("off")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE)

    fig.suptitle("SHAP — Importância Média |SHAP| (conjunto de teste completo)",
                 color=WHITE, fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "shap_global_mean.png"))


# ------------------------------------------------------------------
def plot_sample_grid(
    images: np.ndarray, shap_maps: np.ndarray, predictions: np.ndarray,
    labels: Optional[pd.DataFrame], output_dir: str,
    n_samples: int = 12, seed: int = 42,
) -> None:
    """Grade: imagem | SHAP signed overlay | |SHAP| acumulado."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), min(n_samples, len(images)), replace=False)

    fig, axes = plt.subplots(len(idx), 3, figsize=(12, 3.5 * len(idx)), facecolor=DARK_BG)
    if len(idx) == 1:
        axes = axes[np.newaxis, :]

    for col, ttl in enumerate(["Imagem (Ch 0)", "SHAP Overlay (signed)", "|SHAP| soma canais"]):
        axes[0, col].set_title(ttl, color=WHITE, fontsize=10, pad=8)

    for row, i in enumerate(idx):
        img  = normalize_display(images[i])
        sv   = shap_maps[i]
        sv_s = sv.sum(-1)  # signed sum
        vmax = float(np.percentile(np.abs(sv_s), 99)) or 1e-6
        pred = predictions[i]
        info = f"Pred {pred:.1f} kt"
        if labels is not None and "Vmax" in labels.columns:
            tv = labels["Vmax"].iloc[i]
            info += f" | True {tv:.1f} kt | Err {abs(pred-tv):.1f}"

        ax0, ax1, ax2 = axes[row]
        for ax in (ax0, ax1, ax2):
            ax.set_facecolor(DARK_AX)
            ax.axis("off")

        ax0.imshow(img[..., 0], cmap="gray")
        ax0.set_ylabel(info, color=WHITE, fontsize=7, rotation=0, ha="right", va="center")

        ax1.imshow(img[..., 0], cmap="gray", alpha=0.55)
        im1 = ax1.imshow(sv_s, cmap="seismic", vmin=-vmax, vmax=vmax, alpha=0.6)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        im2 = ax2.imshow(np.abs(sv_s), cmap="hot", vmin=0, vmax=vmax)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("SHAP — Amostras com Rotation Blending TTA", color=WHITE, fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "shap_samples_grid.png"))


# ------------------------------------------------------------------
def plot_error_analysis(
    shap_maps: np.ndarray, predictions: np.ndarray,
    labels: pd.DataFrame, output_dir: str,
) -> None:
    """Scatter |SHAP| vs erro + correlação espacial centro/borda."""
    if labels is None or "Vmax" not in labels.columns:
        return

    true_v  = labels["Vmax"].values
    errors  = np.abs(predictions - true_v)
    shap_mag = np.mean(np.abs(shap_maps), axis=(1, 2, 3))

    H = shap_maps.shape[1]
    s, e = H // 4, 3 * H // 4
    shap_center = np.mean(np.abs(shap_maps[:, s:e, s:e, :]), axis=(1, 2, 3))
    shap_border = shap_mag - shap_center

    corr, pval = spearmanr(shap_mag, errors)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK_BG)
    for ax in axes:
        ax.set_facecolor(DARK_AX)

    # — Scatter SHAP magnitude vs erro —
    ax1 = axes[0]
    sc = ax1.scatter(shap_mag, errors, c=true_v, cmap="plasma", alpha=0.35, s=8)
    ax1.set_xlabel("|SHAP| médio", color=WHITE)
    ax1.set_ylabel("|Erro| (kt)", color=WHITE)
    ax1.set_title(f"Magnitude SHAP × Erro\nSpearman r={corr:.3f}  p={pval:.2e}", color=WHITE)
    ax1.tick_params(colors=WHITE)
    cb = fig.colorbar(sc, ax=ax1)
    cb.set_label("Vmax real (kt)", color=WHITE)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE)

    # — Centro vs Borda por faixa de erro —
    ax2 = axes[1]
    bins = np.histogram_bin_edges(errors, bins=8)
    bc = 0.5 * (bins[:-1] + bins[1:])
    w  = (bins[1] - bins[0]) * 0.38
    dig = np.digitize(errors, bins)
    cm_ = [shap_center[dig == k+1].mean() if (dig == k+1).any() else np.nan for k in range(len(bc))]
    bm_ = [shap_border[dig == k+1].mean() if (dig == k+1).any() else np.nan for k in range(len(bc))]
    cm_ = np.nan_to_num(cm_)
    bm_ = np.nan_to_num(bm_)

    ax2.bar(bc - w/2, cm_, width=w, color="#4e9af1", label="Centro (25–75%)", alpha=0.85)
    ax2.bar(bc + w/2, bm_, width=w, color="#f1724e", label="Borda", alpha=0.85)
    ax2.set_xlabel("|Erro| (kt)", color=WHITE)
    ax2.set_ylabel("SHAP médio por região", color=WHITE)
    ax2.set_title("Concentração SHAP: Centro vs. Borda", color=WHITE)
    ax2.tick_params(colors=WHITE)
    ax2.legend(facecolor=DARK_BG, labelcolor=WHITE)

    mae  = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    fig.suptitle(f"SHAP — Análise de Erro  (MAE={mae:.2f} kt  RMSE={rmse:.2f} kt)",
                 color=WHITE, fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "shap_error_analysis.png"))


# ------------------------------------------------------------------
def plot_prediction_scatter(
    predictions: np.ndarray, labels: pd.DataFrame, output_dir: str,
) -> None:
    """Scatter Vmax_real vs Vmax_predito + histograma de erros."""
    if labels is None or "Vmax" not in labels.columns:
        return
    true_v = labels["Vmax"].values
    errors = predictions - true_v
    mae    = np.mean(np.abs(errors))
    rmse   = np.sqrt(np.mean(errors**2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)
    for ax in axes:
        ax.set_facecolor(DARK_AX)

    ax1 = axes[0]
    sc = ax1.scatter(true_v, predictions, c=np.abs(errors), cmap="plasma",
                     alpha=0.45, s=10, edgecolors="none")
    # lims = [min(true_v.min(), predictions.min()), max(true_v.max(), predictions.max())]
    lims = [0, max(true_v.max(), predictions.max())]
    ax1.plot(lims, lims, "w--", alpha=0.6, lw=1.5, label="Perfeito")
    ax1.set_xlabel("Vmax real (kt)", color=WHITE)
    ax1.set_ylabel("Vmax predito (kt)", color=WHITE)
    ax1.set_title(f"Real × Predito\nMAE={mae:.2f} kt | RMSE={rmse:.2f} kt", color=WHITE)
    ax1.tick_params(colors=WHITE)
    ax1.legend(facecolor=DARK_BG, labelcolor=WHITE)
    cb = fig.colorbar(sc, ax=ax1)
    cb.set_label("|Erro| (kt)", color=WHITE)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE)

    ax2 = axes[1]
    ax2.hist(errors, bins=50, color="#7b2d8b", edgecolor="#b06ec7", alpha=0.85)
    ax2.axvline(0, color=WHITE, ls="--", lw=1.5)
    ax2.set_xlabel("Erro predito−real (kt)", color=WHITE)
    ax2.set_ylabel("Frequência", color=WHITE)
    ax2.set_title("Distribuição dos Erros", color=WHITE)
    ax2.tick_params(colors=WHITE)

    plt.tight_layout()
    _save(fig, os.path.join(output_dir, "shap_prediction_scatter.png"))


# ============================================================================
# Pipeline principal
# ============================================================================

def main(
    n_background: int = 100,
    batch_size:   int = 8,
    rotations:    int = 10,
    limit:        Optional[int] = None,
    n_plot:       int = 12,
    seed:         int = 42,
) -> None:
    sep = "=" * 70
    print(sep)
    print("  SHAP + Rotation Blending — Regressão de Vmax (ciclones)")
    print(sep)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # — 1. Carregar modelo —
    print(f"\nCarregando modelo: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print(f"  Input : {model.input_shape}")
    print(f"  Output: {model.output_shape}")

    # — 2. Carregar dataset —
    print(f"\nCarregando dataset: {DATASET_PATH}")
    with h5py.File(DATASET_PATH, "r") as f:
        images = f["matrix"][:]
    print(f"  Imagens: {images.shape}  dtype={images.dtype}")

    try:
        labels = pd.read_hdf(DATASET_PATH, key="info")
        print(f"  Labels : {labels.shape}  colunas={labels.columns.tolist()}")
    except Exception as exc:
        labels = None
        print(f"  Aviso: não foi possível carregar 'info' — {exc}")

    # — 3. Pré-processamento —
    model_h = model.input_shape[1]
    images = center_crop(images, size=model_h)

    if limit is not None:
        print(f"\n  [LIMIT] Usando as primeiras {limit} amostras.")
        images = images[:limit]
        if labels is not None:
            labels = labels.iloc[:limit].reset_index(drop=True)

    N = len(images)

    # — 4. Background para GradientExplainer —
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(N, min(n_background, N), replace=False)
    background = images[bg_idx]
    print(f"\n  Background set: {background.shape}")

    # — 5. Calcular SHAP com TTA —
    explainer = SHAPRegressionTTA(model, background, rotations=rotations)
    shap_maps, predictions = explainer.process_all(images, batch_size=batch_size)
    print(f"\n  SHAP maps : {shap_maps.shape}  dtype={shap_maps.dtype}")
    print(f"  Predições : {predictions.shape}  min={predictions.min():.1f}  max={predictions.max():.1f}")

    # — 6. Salvar HDF5 —
    out_h5 = os.path.join(OUTPUT_DIR, "shap_results.h5")
    print(f"\nSalvando resultados em: {out_h5}")
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("images",      data=images,      compression="gzip")
        f.create_dataset("shap_values", data=shap_maps,   compression="gzip")
        f.create_dataset("predictions", data=predictions, compression="gzip")
        f.attrs["rotations"]   = rotations
        f.attrs["model_path"]  = MODEL_PATH
        f.attrs["n_samples"]   = N
        f.attrs["n_background"] = n_background
    if labels is not None:
        try:
            labels.to_hdf(out_h5, key="info", mode="a", format="table")
        except Exception as exc:
            print(f"  Aviso ao salvar info: {exc}")
    print("  HDF5 salvo.")

    # — 7. Métricas —
    if labels is not None and "Vmax" in labels.columns:
        true_v = labels["Vmax"].values
        mae    = np.mean(np.abs(predictions - true_v))
        rmse   = np.sqrt(np.mean((predictions - true_v) ** 2))
        print(f"\n  MAE={mae:.2f} kt  |  RMSE={rmse:.2f} kt  (média {rotations} rotações)")

    # — 8. Visualizações —
    print(f"\nGerando plots em: {OUTPUT_DIR}")
    plot_global_mean(shap_maps, OUTPUT_DIR)
    plot_sample_grid(images, shap_maps, predictions, labels, OUTPUT_DIR, n_samples=n_plot, seed=seed)
    plot_error_analysis(shap_maps, predictions, labels, OUTPUT_DIR)
    plot_prediction_scatter(predictions, labels, OUTPUT_DIR)

    print(f"\n{sep}")
    print("  Concluído!")
    print(f"  Output HDF5 : {out_h5}")
    print(f"  Plots       : {OUTPUT_DIR}/")
    print(sep)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SHAP + Rotation Blending para regressão de Vmax de ciclones",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit",      type=int, default=None, help="Limitar a N amostras")
    parser.add_argument("--bg",         type=int, default=100,  help="Tamanho do background set")
    parser.add_argument("--rotations",  type=int, default=10,   help="Número de rotações TTA")
    parser.add_argument("--batch",      type=int, default=8,    help="Batch size para SHAP")
    parser.add_argument("--n-plot",     type=int, default=12,   help="Amostras na grade de plots")
    parser.add_argument("--seed",       type=int, default=42,   help="Seed aleatório")

    args = parser.parse_args()

    main(
        n_background=args.bg,
        batch_size=args.batch,
        rotations=args.rotations,
        limit=args.limit,
        n_plot=args.n_plot,
        seed=args.seed,
    )
