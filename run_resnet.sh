#!/bin/bash
# Script para rodar o treinamento da ResNet com GPU habilitada
# Configura o LD_LIBRARY_PATH com as libs CUDA instaladas via pip

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define a raiz do projeto como variável de ambiente
export RESULT_PATH="$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
SITE_PKGS="$VENV_DIR/lib/python3.12/site-packages"
NVIDIA_DIR="$SITE_PKGS/nvidia"

# Construir LD_LIBRARY_PATH com todos os diretórios de libs nvidia
CUDA_LIBS=""
for pkg_dir in "$NVIDIA_DIR"/*/lib; do
    if [ -d "$pkg_dir" ]; then
        CUDA_LIBS="$pkg_dir:$CUDA_LIBS"
    fi
done

export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Desabilita CUDA command buffers para evitar OOM ao instanciar XLA graphs
# (problema frequente com GPUs Blackwell/RTX 5000 series no TF 2.21)
export XLA_FLAGS="--xla_gpu_enable_command_buffer="

# Força alocação incremental de VRAM (backup do set_memory_growth em training.py)
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "=== Configuração GPU ==="
echo "LD_LIBRARY_PATH configurado com libs CUDA do .venv"
echo "RESULT_PATH: $RESULT_PATH"
echo "XLA_FLAGS: command buffers desabilitados"
echo "TF_FORCE_GPU_ALLOW_GROWTH: true"
echo ""

# Ativar o virtualenv e rodar o treinamento
source "$VENV_DIR/bin/activate"

# Verificar se a GPU está disponível
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✓ GPU detectada: {gpus}')
else:
    print('✗ GPU não detectada! Verifique as dependências.')
    exit(1)
" 2>&1 | grep -E "^[✓✗]"

echo ""
echo "=== Iniciando treinamento ==="
python3 models/resnet/training.py "$@"