# cyclone_models

Código do TCC **"Predição da velocidade de ciclones tropicais utilizando Redes Neurais Convolucionais de baixa e alta complexidade com explicabilidade baseada em SHAP"**.

O projeto prediz a velocidade máxima do vento (`Vmax`, em nós) de ciclones tropicais a partir de imagens de satélite multiespectrais (base TCIR). São comparadas 5 variações de modelo: o CNN-TC original, duas variações com canais temporais derivados (Δ1 e Δ2) e duas arquiteturas pré-treinadas (ResNet50 e MobileNetV2). A explicabilidade é feita com SHAP.

## Dataset (TCIR)

3 arquivos HDF5 (não versionados — ficam em `data/`):

| Arquivo | Conteúdo |
|---|---|
| `TCIR-ATLN_EPAC_WPAC.h5` | Atlântico, Pacífico Leste e Oeste |
| `TCIR-ALL_2017.h5` | Todas as bacias, ano de 2017 |
| `TCIR-CPAC_IO_SH.h5` | Pacífico Central, Índico e Hemisfério Sul |

- Cada arquivo tem `matrix` (imagens `201×201×4`) e `info` (metadados: `ID`, `time`, `Vmax`, lon/lat, etc.).
- **Canais usados:** 0 (infravermelho) e 3 (micro-ondas passivo).
- **Split temporal:** treino 2003–2014, validação 2015–2016, teste 2017.

## Modelos

Cada pasta em `models/` tem `data.py` (pré-processamento) e `training.py` (arquitetura + treino).

| Pasta | Nome no TCC | Canal extra | Entrada | Arquitetura |
|---|---|---|---|---|
| `original` | Original (CNN-TC) | — | 64×64×2 | 4×Conv2D (16→32→64→128) + Dense(256→64→1) |
| `tcc_I` | CNN-TC-Δ1 | Δ1 (1ª diferença) | 64×64×3 | mesma do CNN-TC |
| `tcc_II` | CNN-TC-Δ2 | Δ2 (2ª diferença) | 64×64×3 | mesma do CNN-TC |
| `resnet` | ResNet50 | Δ2 | 224×224×3 | ResNet50 (ImageNet) + Dense(512→64→1) |
| `mobilenet_v2` | MobileNetV2 | Δ2 | 224×224×3 | MobileNetV2 (ImageNet) + Dense(512→64→1) |
| `optical_flow` | (experimental) | fluxo óptico (Farneback) | 64×64 | mesma do CNN-TC |

O canal extra é calculado sobre o canal 0, a partir das diferenças entre imagens consecutivas do mesmo ciclone (ordenadas por tempo):

- **Δ1** = `abs(Iₙ − Iₙ₋₁)` — descarta a 1ª imagem de cada ciclone.
- **Δ2** = `abs(Iₙ − 2·Iₙ₋₁ + Iₙ₋₂)` — descarta as 2 primeiras imagens de cada ciclone.

## Pipeline de pré-processamento (`models/*/data.py`)

1. Carrega e concatena os 3 HDF5 (imagens lazy via dask).
2. Separa treino/validação/teste por ano.
3. Recorta no centro para `92×92` (margem para rotação sem cantos pretos) e limpa NaN / valores `> 1000`.
4. Normaliza os canais 0 e 3 pela média/desvio do **treino**.
5. (Δ1/Δ2/fluxo óptico) cria e concatena o canal extra.
6. `save_preprocessed()` grava `data/preprocessed/{train,valid,test}.h5`.

No treino, cada imagem recebe rotação aleatória (data augmentation) e recorte central para o tamanho de entrada.

## Estrutura do repositório

```
.
├── models/                        # 1 pasta por modelo (data.py + training.py)
│   ├── original/                  # CNN-TC baseline
│   ├── tcc_I/                     # CNN-TC + canal Δ1
│   ├── tcc_II/                    # CNN-TC + canal Δ2
│   ├── resnet/                    # ResNet50 + Δ2
│   ├── mobilenet_v2/              # MobileNetV2 + Δ2
│   └── optical_flow/              # variação experimental (fluxo óptico)
│
├── analytics/                     # estatística e distribuições (rodam local, leem CSV)
│   ├── wilcoxon_test.py           # Tabelas 3 e 4: RMSE/MAE/MSE + Wilcoxon (Original vs demais)
│   ├── shapiro_test.py            # Tabela 5: normalidade (Shapiro-Wilk)
│   ├── model_error_analysis.py    # distribuição dos erros absolutos por modelo + error_stats.json
│   ├── statistical_tests.py       # homogeneidade dos splits (Kruskal-Wallis + KS)
│   └── plot_distributions.py      # distribuição de Vmax por split (hist/box/violin)
│
├── plot/                          # geração de figuras
│   ├── plot_result.py             # scatter predito×real com TTA (rotation blending)
│   ├── plot_training.py           # curvas de loss treino/validação dos 5 modelos (history.pkl)
│   ├── plot_convolution.py        # figura didática: como funciona a convolução
│   ├── plot_new_channel.py        # figura didática: criação do novo canal (DTA/SDTA)
│   ├── visualize_tcc_ii_preprocessing.py  # ilustra a dupla diferença (Δ2)
│   ├── extract_channels.py        # exporta cada canal de uma imagem TCIR como PNG
│   └── opticalflow.py             # visualização de fluxo óptico (Farneback)
│
├── shap_explainer.py              # mapas SHAP (GradientExplainer) do modelo tcc_II
├── gradcam.py                     # mapas Grad-CAM a partir de um modelo treinado
├── inspec.py                      # visualizador interativo das imagens TCIR brutas
├── inspecpreprocessed.py          # visualizador interativo do train.h5 pré-processado
│
├── cyclone.ipynb                  # orquestração de treino no Kaggle (env vars + data/training)
├── kaggle_generate_results.ipynb  # gera TODOS os resultados do TCC no Kaggle (ver abaixo)
├── util.ipynb                     # rascunhos (inspeção de history.pkl / errors.csv)
│
├── run_resnet.sh                  # treina ResNet local em máquina x86 + GPU NVIDIA
├── requirements.txt               # dependências (macOS / Apple Silicon)
├── requirementsx86.txt            # dependências (x86 + GPU) — versão enxuta
└── requirementsx86new.txt         # dependências (x86 + GPU) — versão completa
```

> **Não versionados (`.gitignore`):** `data/` (`*.h5`), `result/` (modelos treinados, `history.pkl`, `errors.csv`) e `kaggle_results/` (saídas baixadas do Kaggle: `errors/`, `plots/`, `shap/`).

## Restrição de execução (importante)

Os modelos (`.keras`) só rodam de forma confiável em **x86 + GPU NVIDIA** (treinados no Kaggle T4). No **Mac M1** o carregamento falha ou gera predições inválidas.

| Roda local (Mac M1, sem modelo) | Precisa de x86 + GPU (carrega o modelo) |
|---|---|
| `analytics/*` (leem `errors.csv`) | treino (`models/*/training.py`) |
| `plot/plot_training.py`, `plot_convolution.py`, `plot_new_channel.py`, `plot_distributions.py`, `extract_channels.py`, `visualize_tcc_ii_preprocessing.py` | `plot/plot_result.py` (TTA) |
| `inspec.py`, `inspecpreprocessed.py` | `gradcam.py`, `shap_explainer.py` |

Por isso, inferência e geração de erros são feitas no Kaggle (ver `kaggle_generate_results.ipynb`), e as análises estatísticas/figuras são feitas localmente a partir dos CSVs baixados em `kaggle_results/`.

## Geração de resultados no Kaggle

`kaggle_generate_results.ipynb` reproduz, a partir dos dados brutos TCIR e dos `model.keras`, todas as saídas usadas no TCC:

- `plots/{modelo}_scatter.png` — Tabela 3 (MSE / RMSE / MAE, com TTA de 10 rotações)
- `errors/{modelo}/errors.csv` — base do Wilcoxon e do Shapiro-Wilk (também com TTA)
- `wilcoxon_summary.json` / `shapiro_wilk_summary.json` — Tabelas 4 e 5
- `shap/` — mapas de explicabilidade SHAP do tcc_II

Os erros de todos os modelos cobrem o mesmo subconjunto de amostras (a partir da 3ª imagem de cada ciclone), permitindo o teste pareado de Wilcoxon.

## Como rodar

**Pré-processar dados** (gera `data/preprocessed/`):
```bash
python models/<modelo>/data.py
```

**Treinar** (x86 + GPU; `RESULT_PATH` define onde salvar `model.keras` e `history.pkl`):
```bash
RESULT_PATH=result/<modelo> python models/<modelo>/training.py
# ou, para a ResNet em máquina local com CUDA no .venv:
./run_resnet.sh
```

**Análises locais** (a partir de `kaggle_results/errors/`):
```bash
python analytics/wilcoxon_test.py     # Tabelas 3 e 4
python analytics/shapiro_test.py      # Tabela 5
python analytics/model_error_analysis.py
python plot/plot_training.py          # curvas de loss (lê result/*/history.pkl)
```

## Dependências

- **macOS / Apple Silicon:** `pip install -r requirements.txt` (usa `tensorflow-macos` + `tensorflow-metal`).
- **x86 + GPU NVIDIA:** `pip install -r requirementsx86new.txt`.
