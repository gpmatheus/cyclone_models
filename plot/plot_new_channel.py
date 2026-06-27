"""
Figura didática: criação de um novo canal a partir da sequência temporal do ciclone.

Mostra 6 imagens do MESMO ciclone, ordenadas cronologicamente (uma ao lado da
outra). Cada imagem é representada como uma pilha de planos (canais) levemente
desalinhados, evidenciando que a imagem tem múltiplos canais:

  • canal 0  (cinza, ao fundo, deslocado)
  • canal 3  (cinza, à frente)
  • novo canal criado (plano verde) — presente até a 4ª imagem

Na 5ª imagem há um quadrado com as duas opções de método de criação do novo
canal: DTA e SDTA. Esses métodos correspondem ao que é feito em
models/tcc_I/data.py (compute = |I_n - I_{n-1}|) e na variante de 2ª diferença:

  DTA  : |I_n - I_{n-1}|                (1ª diferença temporal absoluta)
  SDTA : |I_n - 2*I_{n-1} + I_{n-2}|    (2ª diferença temporal absoluta)

O novo canal é gerado a partir da diferença entre frames consecutivos do canal 0.

Uso:
    python plot/plot_new_channel.py

Saída:
    result/plots/criacao_novo_canal.png
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent
H5_PATH  = ROOT_DIR / "data" / "TCIR-ALL_2017.h5"
OUT_PATH = ROOT_DIR / "result" / "plots" / "criacao_novo_canal.png"

CHANNELS    = [0, 3]      # canais reais exibidos (índice 0 e 3)
CYCLONE_ID  = '201707W'   # ciclone exibido
START_REL   = 94          # índice (relativo ao ciclone) do 1º frame mostrado
N_FRAMES    = 6
DISP_CROP   = 120         # recorte central (px) só para exibição

OFFSET      = 0.17        # desalinhamento entre os planos da pilha (fração do lado)
GREEN       = '#2bb24c'   # cor do novo canal
GREEN_DK    = '#1d7a34'
GRAY_BORDER = 'white'

N_WITH_GREEN = 4          # novo canal (verde) aparece até esta imagem (1..N)
METHOD_CARD  = 5          # imagem que recebe o quadro DTA / SDTA (1-based)


# =============================================================================
# LEITURA DOS DADOS
# =============================================================================

def read_meta(h5_path):
    """Retorna (ids, times) alinhados ao matrix. Robusto a versões do pandas."""
    try:
        import pandas as pd
        info = pd.read_hdf(h5_path, key='info', mode='r')
        return info['ID'].to_numpy().astype(str), info['time'].to_numpy().astype(str)
    except Exception:
        import pickle
        with h5py.File(h5_path, 'r') as f:
            g = f['info']
            items = [x.decode() if isinstance(x, bytes) else x
                     for x in g['block1_items'][:]]
            meta = np.array(pickle.loads(g['block1_values'][0].tobytes()))
        col = {name: i for i, name in enumerate(items)}
        return meta[:, col['ID']].astype(str), meta[:, col['time']].astype(str)


def cyclone_window(h5_path, cyclone_id, start_rel, n):
    """Frames de um ciclone em ordem cronológica.

    Retorna (global_idx_window, rel_indices, times_window, n_total), onde
    rel_indices são as posições da imagem DENTRO do ciclone (0..n_total-1).
    """
    ids, times = read_meta(h5_path)
    gidx = np.where(ids == cyclone_id)[0]
    if len(gidx) == 0:
        raise ValueError(f"Ciclone '{cyclone_id}' não encontrado.")
    gidx_sorted = gidx[np.argsort(times[gidx])]   # ordena por tempo
    n_total = len(gidx_sorted)
    end_rel = min(start_rel + n, n_total)
    win = gidx_sorted[start_rel:end_rel]
    rel = list(range(start_rel, end_rel))
    return win, rel, times[win], n_total


def load_frames(h5_path, global_indices):
    with h5py.File(h5_path, 'r') as f:
        imgs = np.stack([f['matrix'][int(i)].astype(np.float32) for i in global_indices])
    imgs = np.nan_to_num(imgs)
    imgs[imgs > 1000] = 0
    return imgs


def crop_center(img2d, size):
    h, w = img2d.shape
    cy, cx = h // 2, w // 2
    hs = size // 2
    return img2d[cy - hs:cy + hs, cx - hs:cx + hs]


def norm01(a):
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-9)


def fmt_time(t):
    """'YYYYMMDDHH' -> 'dd/mm HHh'."""
    try:
        return f"{int(t[6:8]):02d}/{int(t[4:6]):02d} {t[8:10]}h"
    except Exception:
        return t


# =============================================================================
# DESENHO DE UM CARTÃO (pilha de canais desalinhados)
# =============================================================================

def make_green_tile(h=40, w=40):
    """Plano verde com leve gradiente, para parecer uma 'imagem' de canal."""
    grad = np.linspace(0.85, 1.05, h)[:, None]
    rgb = np.zeros((h, w, 3))
    base = np.array([0.17, 0.70, 0.30])
    rgb[:] = np.clip(base[None, None, :] * grad[..., None], 0, 1)
    return rgb


def draw_layer(ax, data, shift, cmap, border, zorder, alpha=1.0, is_rgb=False):
    S = 1.0
    ext = [shift, shift + S, shift, shift + S]
    if is_rgb:
        ax.imshow(data, extent=ext, origin='upper', zorder=zorder, alpha=alpha)
    else:
        ax.imshow(data, cmap=cmap, extent=ext, origin='upper',
                  zorder=zorder, alpha=alpha, vmin=0, vmax=1)
    ax.add_patch(Rectangle((shift, shift), S, S, fill=False,
                           edgecolor=border, linewidth=1.4,
                           zorder=zorder + 0.1, alpha=alpha))


def draw_card(ax, ch0, ch3, with_green, dim=1.0):
    """Desenha a pilha: novo canal (fundo) + canal 0 + canal 3 (frente)."""
    # ordem traseira -> frente: novo canal (fundo), canal 3, canal 0 (frente)
    layers = []
    if with_green:
        layers.append(('green', make_green_tile(), GREEN_DK))
    layers.append(('ch3', ch3, GRAY_BORDER))
    layers.append(('ch0', ch0, GRAY_BORDER))

    n = len(layers)
    for i, (kind, data, border) in enumerate(layers):   # i=0 fundo
        shift = (n - 1 - i) * OFFSET
        if kind == 'green':
            draw_layer(ax, data, shift, None, border, zorder=i, alpha=dim, is_rgb=True)
        else:
            draw_layer(ax, data, shift, 'gray', border, zorder=i, alpha=dim)

    # limites FIXOS (espaço para até 3 planos) — todas as imagens do mesmo tamanho,
    # independente de o cartão ter ou não o plano verde.
    lim_hi = 1.0 + 2 * OFFSET
    ax.set_xlim(-0.06, lim_hi + 0.06)
    ax.set_ylim(-0.06, lim_hi + 0.06)
    ax.set_aspect('equal')
    ax.axis('off')
    return n


def draw_method_box(ax):
    """Quadro com as duas opções de criação do novo canal: DTA e SDTA."""
    box = FancyBboxPatch((0.06, 0.40), 0.88, 0.24,
                         boxstyle="round,pad=0.02,rounding_size=0.04",
                         transform=ax.transAxes, facecolor='white',
                         edgecolor='#333333', linewidth=2, zorder=20)
    ax.add_patch(box)

    for cx, label, color in [(0.31, 'DTA', GREEN), (0.69, 'SDTA', '#2f6fb0')]:
        chip = FancyBboxPatch((cx - 0.17, 0.45), 0.34, 0.14,
                              boxstyle="round,pad=0.01,rounding_size=0.03",
                              transform=ax.transAxes, facecolor=color,
                              edgecolor='none', zorder=21)
        ax.add_patch(chip)
        ax.text(cx, 0.52, label, transform=ax.transAxes, ha='center',
                va='center', fontsize=12, fontweight='bold',
                color='white', zorder=22)


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not H5_PATH.exists():
        print(f"❌ Dataset não encontrado: {H5_PATH}")
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    win, rel_idx, win_times, n_total = cyclone_window(
        H5_PATH, CYCLONE_ID, START_REL, N_FRAMES)
    frames = load_frames(H5_PATH, win)
    print(f"Ciclone {CYCLONE_ID}: {n_total} imagens (índices 0..{n_total - 1})  "
          f"|  exibindo índices {rel_idx[0]}..{rel_idx[-1]}")

    # canais exibidos, recortados e normalizados
    ch0_list, ch3_list = [], []
    for k in range(len(rel_idx)):
        ch0_list.append(norm01(crop_center(frames[k, :, :, CHANNELS[0]], DISP_CROP)))
        ch3_list.append(norm01(crop_center(frames[k, :, :, CHANNELS[1]], DISP_CROP)))

    # ---- figura ----
    fig = plt.figure(figsize=(20, 6.2))
    gs = fig.add_gridspec(1, N_FRAMES, left=0.025, right=0.985,
                          top=0.73, bottom=0.30, wspace=0.12)

    axes = []
    for k in range(N_FRAMES):
        ax = fig.add_subplot(gs[k])
        axes.append(ax)

        is_method = (k + 1) == METHOD_CARD
        with_green = (k + 1) <= N_WITH_GREEN

        if is_method:
            draw_card(ax, ch0_list[k], ch3_list[k], with_green=False)  # sem transparência
            draw_method_box(ax)
        else:
            draw_card(ax, ch0_list[k], ch3_list[k], with_green=with_green)

        # rótulo: índice da imagem (relativo ao ciclone) e data/hora
        ax.set_title(f"índice {rel_idx[k]}\n{fmt_time(win_times[k])}",
                     fontsize=10.5, fontweight='bold', pad=6)

    fig.canvas.draw()  # garante posições das axes

    # ---- seta de tempo (cronologia) acima dos cartões ----
    p_first = axes[0].get_position()
    p_last  = axes[-1].get_position()
    y_arrow = 0.88
    fig.add_artist(FancyArrowPatch(
        (p_first.x0, y_arrow), (p_last.x1, y_arrow),
        transform=fig.transFigure, arrowstyle='-|>', mutation_scale=22,
        lw=2, color='#555555'))
    fig.text((p_first.x0 + p_last.x1) / 2, y_arrow + 0.015,
             'tempo (ordem cronológica)', ha='center', va='bottom',
             fontsize=12, color='#555555', fontweight='bold')

    # ---- fórmulas dos métodos no rodapé ----
    fig.text(0.5, 0.16,
             'DTA:  |$I_n - I_{n-1}$|        '
             'SDTA:  |$I_n - 2\\,I_{n-1} + I_{n-2}$|        '
             '(diferenças entre frames consecutivos do canal 0)',
             ha='center', va='center', fontsize=11.5,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f3f3f3',
                       edgecolor='#999999'))

    fig.suptitle(f'Criação de um novo canal a partir da sequência temporal do ciclone '
                 f'(ID {CYCLONE_ID}, {n_total} imagens)',
                 fontsize=16, fontweight='bold', y=0.99)

    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Figura salva em: {OUT_PATH}")


if __name__ == '__main__':
    main()
