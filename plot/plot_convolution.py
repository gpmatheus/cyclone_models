"""
Figura didática: como funciona a convolução de um kernel sobre uma imagem.

Gera uma única figura que ilustra, da esquerda para a direita:

  1. A imagem real do ciclone (canal IR do TCIR), com um retângulo destacando
     a região que será ampliada.
  2. A região ampliada como uma grade de valores (a "entrada" da convolução),
     com o valor numérico em cada célula e a janela 4x4 do kernel sobreposta
     em duas posições consecutivas — evidenciando o passo (stride = 2).
  3. O kernel 4x4 (o filtro), onde cada peso tem uma cor (mapa divergente:
     azul = negativo, vermelho = positivo) e o valor numérico visível.
  4. O mapa de ativação resultante (a "saída"), também com valores e cores,
     destacando as células geradas pelas duas posições da janela.

A convolução usada é a correlação cruzada (convenção de CNNs):

    saida(i, j) = Σ  entrada[stride*i + a, stride*j + b] * kernel[a, b]
                 a,b

Entrada 8x8, kernel 4x4, stride 2  →  saída 3x3.

Uso:
    python plot/plot_convolution.py

A figura é salva em result/plots/convolucao_kernel.png
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from pathlib import Path

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

ROOT_DIR  = Path(__file__).parent.parent
H5_PATH   = ROOT_DIR / "data" / "TCIR-ALL_2017.h5"
OUT_PATH  = ROOT_DIR / "result" / "plots" / "convolucao_kernel.png"

CHANNEL   = 0       # canal 0 = imagem infravermelha (IR)
IMG_INDEX = 2075    # ciclone com olho visível (alternativas boas: 3300, 2325)

CROP      = 120     # tamanho do recorte central (deve ser múltiplo de GRID_IN)
GRID_IN   = 8       # lado da grade de entrada (região ampliada)
STRIDE    = 2       # passo do kernel

# Kernel 4x4 — detector de borda vertical (esquerda escura, direita clara).
# Pesos negativos (azul) e positivos (vermelho) deixam o mapa de cor expressivo.
KERNEL = np.array([
    [-1, -1, 1, 1],
    [-1, -1, 1, 1],
    [-1, -1, 1, 1],
    [-1, -1, 1, 1],
], dtype=float)

# Mapas de cor
CMAP_IN  = 'gray'      # entrada (fiel à imagem IR)
CMAP_KER = 'coolwarm'  # kernel (divergente: sinal do peso)
CMAP_OUT = 'coolwarm'  # saída  (divergente: sinal da ativação)

# Cores de destaque das duas posições da janela (stride)
COR_POS0 = '#00c853'   # verde  — posição inicial  → saída(0,0)
COR_POS1 = '#ff6d00'   # laranja — após 1 passo     → saída(0,1)


# =============================================================================
# CARGA E PREPARO DOS DADOS
# =============================================================================

def load_cyclone(h5_path, index, channel):
    """Carrega uma imagem de ciclone do TCIR e limpa NaN / valores inválidos."""
    with h5py.File(h5_path, mode='r') as f:
        img = f['matrix'][index, :, :, channel].astype(np.float32)
    img = np.nan_to_num(img)
    img[img > 1000] = 0
    return img


def extract_patch(img, crop, grid):
    """Recorta a região central e reduz para grid x grid por média de blocos.

    Usar média de blocos (em vez de pixels adjacentes) faz a grade representar
    de fato a estrutura do ciclone (olho quente + parede fria), com variação
    visível entre as células.
    """
    h, w = img.shape
    cy, cx = h // 2, w // 2
    half = crop // 2
    region = img[cy - half: cy + half, cx - half: cx + half]

    block = crop // grid
    patch = region.reshape(grid, block, grid, block).mean(axis=(1, 3))
    return region, patch


def quantize(patch, levels=9):
    """Normaliza a região para inteiros [0, levels] (mais legível na figura)."""
    p = patch - patch.min()
    if p.max() > 0:
        p = p / p.max()
    return np.round(p * levels).astype(int).astype(float)


def convolve2d_stride(x, k, stride):
    """Correlação cruzada (convenção CNN) com passo `stride`. Saída 2D."""
    kh, kw = k.shape
    oh = (x.shape[0] - kh) // stride + 1
    ow = (x.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow), dtype=float)
    for i in range(oh):
        for j in range(ow):
            r0, c0 = i * stride, j * stride
            out[i, j] = np.sum(x[r0:r0 + kh, c0:c0 + kw] * k)
    return out


# =============================================================================
# DESENHO
# =============================================================================

def draw_value_grid(ax, values, cmap, title, fmt='{:.0f}',
                    fontsize=11, divergent=False, vmin=None, vmax=None):
    """Desenha uma matriz como grade colorida com o valor numérico em cada célula."""
    if divergent:
        lim = np.abs(values).max() or 1.0
        vmin, vmax = -lim, lim

    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    nr, nc = values.shape

    # linhas da grade
    ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='both', length=0)
    ax.set_xticks([]); ax.set_yticks([])

    # texto com contraste automático
    for r in range(nr):
        for c in range(nc):
            rgba = im.cmap(im.norm(values[r, c]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(c, r, fmt.format(values[r, c]), ha='center', va='center',
                    color=('white' if lum < 0.5 else 'black'),
                    fontsize=fontsize, fontweight='bold')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    return im


def add_window(ax, r0, c0, size, color, dashed=False, lw=3):
    """Marca uma janela quadrada (do kernel) sobre a grade de entrada."""
    rect = Rectangle((c0 - 0.5, r0 - 0.5), size, size, fill=False,
                     edgecolor=color, linewidth=lw,
                     linestyle=('--' if dashed else '-'), zorder=5)
    ax.add_patch(rect)


def add_cell_border(ax, r, c, color, dashed=False, lw=3):
    """Contorna uma célula da grade de saída."""
    rect = Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                     edgecolor=color, linewidth=lw,
                     linestyle=('--' if dashed else '-'), zorder=5)
    ax.add_patch(rect)


def main():
    if not H5_PATH.exists():
        print(f"❌ Dataset não encontrado: {H5_PATH}")
        print("   Ajuste H5_PATH no topo do script.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---- dados ----
    img = load_cyclone(H5_PATH, IMG_INDEX, CHANNEL)
    region, patch_raw = extract_patch(img, CROP, GRID_IN)
    patch = quantize(patch_raw, levels=9)
    out = convolve2d_stride(patch, KERNEL, STRIDE)

    kh, kw = KERNEL.shape
    # janelas das duas primeiras posições de saída (mesma linha)
    win0 = (0, 0)                 # → saída(0,0)
    win1 = (0, STRIDE)            # → saída(0,1)
    val0 = out[0, 0]
    val1 = out[0, 1]

    # ---- figura ----
    fig = plt.figure(figsize=(18, 7.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.35, 0.75, 0.95],
                          wspace=0.28, left=0.04, right=0.985,
                          top=0.86, bottom=0.16)
    ax_cyc = fig.add_subplot(gs[0])
    ax_in  = fig.add_subplot(gs[1])
    ax_ker = fig.add_subplot(gs[2])
    ax_out = fig.add_subplot(gs[3])

    # 1) imagem do ciclone + retângulo da região ampliada
    ax_cyc.imshow(img, cmap='gray')
    ax_cyc.set_title('Imagem do ciclone\n(canal IR)', fontsize=13, fontweight='bold', pad=10)
    ax_cyc.axis('off')
    h, w = img.shape
    half = CROP // 2
    ax_cyc.add_patch(Rectangle((w // 2 - half, h // 2 - half), CROP, CROP,
                               fill=False, edgecolor='red', linewidth=2.5))

    # 2) entrada (região ampliada) com valores + janelas do kernel (stride)
    im_in = draw_value_grid(ax_in, patch, CMAP_IN,
                            'Região ampliada (entrada)', fontsize=12)
    add_window(ax_in, win0[0], win0[1], kh, COR_POS0, dashed=False)
    add_window(ax_in, win1[0], win1[1], kh, COR_POS1, dashed=True)

    # seta indicando o passo (stride) entre as duas janelas — abaixo da grade
    y_arrow = GRID_IN - 0.1
    ax_in.annotate(
        '', xy=(win1[1] + (kw - 1) / 2, y_arrow),
        xytext=(win0[1] + (kw - 1) / 2, y_arrow),
        arrowprops=dict(arrowstyle='<->', color='black', lw=2),
        annotation_clip=False,
    )
    ax_in.text((win0[1] + win1[1] + (kw - 1)) / 2, y_arrow + 0.45,
               f'passo (stride) = {STRIDE}', ha='center', va='top',
               fontsize=11, fontweight='bold')

    # 3) kernel
    draw_value_grid(ax_ker, KERNEL, CMAP_KER, 'Kernel 4×4\n(filtro)',
                    fontsize=15, divergent=True)
    ax_ker.text(0.5, -0.16, 'azul = peso negativo\nvermelho = peso positivo',
                transform=ax_ker.transAxes, ha='center', va='top', fontsize=9)

    # 4) saída (mapa de ativação)
    draw_value_grid(ax_out, out, CMAP_OUT, 'Mapa de ativação\n(saída)',
                    fontsize=14, divergent=True)
    add_cell_border(ax_out, 0, 0, COR_POS0, dashed=False)
    add_cell_border(ax_out, 0, 1, COR_POS1, dashed=True)

    # ---- conexões entre painéis ----
    # ciclone (região) → entrada
    con = ConnectionPatch(
        xyA=(w // 2 + half, h // 2), coordsA=ax_cyc.transData,
        xyB=(-0.5, (GRID_IN - 1) / 2), coordsB=ax_in.transData,
        arrowstyle='->', color='red', lw=2, mutation_scale=18,
    )
    fig.add_artist(con)

    # ---- faixa de fórmula / explicação ----
    soma_txt = (
        f'Saída(0,0) = Σ [ janela verde ⊙ kernel ] = {val0:.0f}        '
        f'avança o passo (stride={STRIDE})  →   '
        f'Saída(0,1) = Σ [ janela laranja ⊙ kernel ] = {val1:.0f}'
    )
    fig.text(0.5, 0.045, soma_txt, ha='center', va='center', fontsize=12.5,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f3f3f3',
                       edgecolor='#999999'))

    fig.suptitle('Convolução de um kernel sobre a imagem do ciclone',
                 fontsize=16, fontweight='bold', y=0.97)

    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Figura salva em: {OUT_PATH}")


if __name__ == '__main__':
    main()
