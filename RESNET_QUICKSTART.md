# 🚀 Quick Start: Como Usar ResNet152 no Seu Projeto

## 1️⃣ Instalação de Dependências

Certifique-se de que tem TensorFlow 2.8+ com suporte a ResNet152:

```bash
# Se ainda não tem, instale:
pip install tensorflow>=2.8 keras>=2.8

# Ou se usa conda:
conda install tensorflow>=2.8 -c conda-forge
```

## 2️⃣ Estrutura de Pastas Atual

```
cyclone_models/
├── models/
│   ├── tcc_II/              ← Original (CNN customizada)
│   │   ├── data.py
│   │   └── training.py
│   │
│   └── resnet/              ← Novo (ResNet152 Transfer Learning) 🆕
│       ├── data.py          (mesma que tcc_II)
│       └── training.py      (com ResNet152)
│
├── data/
│   ├── TCIR-*.h5
│   └── preprocessed/
│       ├── train.h5
│       ├── valid.h5
│       └── test.h5
│
├── result/
│   ├── original/
│   ├── tcc_I/
│   ├── tcc_II/
│   └── resnet/              ← Modelos salvos aqui 🆕
│
├── TRANSFER_LEARNING_GUIDE.md       (teoria detalhada)
├── RESNET_IMPLEMENTATION_SUMMARY.md (resumo)
└── RESNET_QUICKSTART.md             (este arquivo)
```

## 3️⃣ Executar o Treinamento

### 📍 Opção A: Transfer Learning (Recomendado - Rápido)

```python
#!/usr/bin/env python3
"""
Treinar ResNet152 com Transfer Learning
Apenas o head é treinado (ResNet152 congelada)
"""

import os
import sys

# Configurar variáveis de ambiente
os.environ['RESULT_PATH'] = './result/resnet'
os.environ['DATA_PATH'] = './data'
os.environ['PREPROCESSED_PATH'] = './data/preprocessed'

# Adicionar ao path
sys.path.insert(0, './models')

# Importar o módulo de treinamento
from resnet import training

# Executar treinamento
training.main(
    channels=[0, 3],              # Use os mesmos canais de tcc_II
    generated_channels=[0],       # Use os mesmos gerados
    img_w=64,                     # Será redimensionado para 224x224 internamente
    batch=8,                      # Batch size
    learning_rate=5e-5,           # Taxa de aprendizado
    epochs=100,                   # Máximo de epochs (early stopping vai parar antes)
    sample_pct=1.0,               # Use 100% dos dados
    seed=42,                      # Para reprodutibilidade
    patience=30,                  # Early stopping: 30 epochs sem melhora
    l2_regularizer=1e-5,          # Regularização L2
    force=True,                   # Reprocessar dados
    freeze_base=True              # ⭐ TRANSFER LEARNING: congela ResNet152
)

print("✅ Treinamento finalizado!")
print("📁 Modelo salvo em: ./result/resnet/model.keras")
```

**Tempo esperado**: 2-3 horas (GPU) ou 8-12 horas (CPU)

---

### 📍 Opção B: Fine-tuning (Depois de Transfer Learning)

```python
#!/usr/bin/env python3
"""
Fine-tuning: Descongelamos ResNet152 e ajustamos TODA a rede
Execute APÓS Transfer Learning convergir (loss estabilizou)
"""

import os
import sys

os.environ['RESULT_PATH'] = './result/resnet'
os.environ['DATA_PATH'] = './data'
os.environ['PREPROCESSED_PATH'] = './data/preprocessed'

sys.path.insert(0, './models')

from resnet import training

# Fine-tuning com aprendizado MUITO baixo
training.main(
    channels=[0, 3],
    generated_channels=[0],
    img_w=64,
    batch=4,                      # Batch MENOR (mais memória necessária)
    learning_rate=1e-7,           # ⚠️ Taxa MUITO MENOR que Transfer Learning
    epochs=30,                    # Menos epochs necessários
    sample_pct=1.0,
    seed=42,
    patience=15,
    l2_regularizer=1e-5,
    force=False,                  # Reutilizar dados já processados
    freeze_base=False             # ⭐ FINE-TUNING: descongelamos ResNet152
)

print("✅ Fine-tuning finalizado!")
print("🎯 Melhora esperada: +5-10% em performance")
```

**Tempo esperado**: 1-2 horas (GPU) ou 4-6 horas (CPU)

---

### 📍 Opção C: Comparar com tcc_II Original

```python
#!/usr/bin/env python3
"""
Comparar modelos: ResNet vs CNN customizada
"""

import os
import sys
import tensorflow as tf
from pathlib import Path

# Carregar ambos os modelos
model_tcc2 = tf.keras.models.load_model('./result/tcc_II/model.keras')
model_resnet = tf.keras.models.load_model('./result/resnet/model.keras')

print("=" * 60)
print("COMPARAÇÃO: TCC_II (CNN) vs RESNET (Transfer Learning)")
print("=" * 60)

# Contar parâmetros
tcc2_params = model_tcc2.count_params()
resnet_params = model_resnet.count_params()

print(f"\n📊 PARÂMETROS TREINÁVEIS:")
print(f"   tcc_II (CNN):        {tcc2_params:,}")
print(f"   ResNet152:           {resnet_params:,}")
print(f"   Diferença:           {abs(resnet_params - tcc2_params):,}")

# Tamanho dos arquivos
tcc2_size = Path('./result/tcc_II/model.keras').stat().st_size / 1e6
resnet_size = Path('./result/resnet/model.keras').stat().st_size / 1e6

print(f"\n💾 TAMANHO DO MODELO:")
print(f"   tcc_II:    {tcc2_size:.1f} MB")
print(f"   ResNet:    {resnet_size:.1f} MB")

# Resumos das arquiteturas
print(f"\n🏗️  ARQUITETURA TCC_II:")
model_tcc2.summary()

print(f"\n🏗️  ARQUITETURA RESNET:")
model_resnet.summary()
```

---

### 📍 Opção D: Fazer Predições

```python
#!/usr/bin/env python3
"""
Usar modelo treinado para fazer predições
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

# Carregar modelo
model = tf.keras.models.load_model('./result/resnet/model.keras')

# Exemplos de predição
def predict_cyclone_velocity(image_array):
    """
    Predizer velocidade de ciclone
    
    Args:
        image_array: numpy array (H, W, C) com a imagem
    
    Returns:
        Vmax_predito: velocidade em nós
    """
    
    # Pré-processamento
    image = tf.cast(image_array, tf.float32)
    image = tf.image.resize(image, (224, 224))  # Redimensionar para 224x224
    
    # Garantir 3 canais (RGB)
    if image.shape[-1] == 1:
        image = tf.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 2:
        zero_channel = tf.zeros_like(image[:, :, :1])
        image = tf.concat([image, zero_channel], axis=-1)
    
    # Batch (adicionar dimensão batch)
    image = tf.expand_dims(image, 0)
    
    # Predição
    vmax_pred = model.predict(image, verbose=0)
    
    return float(vmax_pred[0][0])


# Teste
if __name__ == "__main__":
    # Imagem fake para teste
    fake_image = np.random.randn(64, 64, 2).astype(np.float32)
    
    vmax = predict_cyclone_velocity(fake_image)
    print(f"Velocidade predita: {vmax:.2f} nós")
```

---

## 4️⃣ Monitorar o Treinamento

### Visualizar Loss em Tempo Real

```python
#!/usr/bin/env python3
"""
Monitorar treinamento com TensorBoard
"""

# Adicionar durante treinamento em training.main():
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs/resnet',
    histogram_freq=1,
    profile_batch='500,520'
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr, tensorboard_callback]
)

# Depois, visualizar:
# tensorboard --logdir ./logs/resnet
# Acesse: http://localhost:6006
```

### Plotar Loss e Métricas

```python
import matplotlib.pyplot as plt
import pickle

# Carregar histórico
with open('./result/resnet/history.pkl', 'rb') as f:
    history = pickle.load(f)

# Plotar
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss
axes[0].plot(history['loss'], label='Train Loss')
axes[0].plot(history['val_loss'], label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE')
axes[0].set_title('Loss (ResNet152)')
axes[0].legend()
axes[0].grid()

# MAE
axes[1].plot(history['mae'], label='Train MAE')
axes[1].plot(history['val_mae'], label='Val MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE (nós)')
axes[1].set_title('Mean Absolute Error')
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.savefig('./result/resnet/training_history.png', dpi=150)
plt.show()

print(f"Melhor MAE alcançado: {min(history['val_mae']):.2f} nós")
```

---

## 5️⃣ Troubleshooting

### ❌ Problema: "ModuleNotFoundError: No module named 'tensorflow'"

```bash
pip install tensorflow>=2.8
```

### ❌ Problema: "ResNet152 não baixa (conexão)"

```python
# Baixar pesos manualmente
import tensorflow as tf
model = tf.keras.applications.ResNet152(weights='imagenet')
# Cacheia em ~/.keras/models/
```

### ❌ Problema: "Out of Memory (GPU)"

```python
# Reduzir batch size
training.main(..., batch=4)  # Era 8, agora 4

# Ou desligar deterministic mode (mais rápido)
# Remove: tf.config.experimental.enable_op_determinism()
```

### ❌ Problema: "Treinamento muito lento (CPU)"

```python
# Use GPU:
# 1. Instalar CUDA e cuDNN
# 2. TensorFlow detectará automaticamente

# Verificar:
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## 6️⃣ Estrutura de Output

Após treinamento, a pasta `./result/resnet/` conterá:

```
result/resnet/
├── model.keras           # Modelo completo (TF SavedModel format)
├── model/                # Pasta com modelo exportado
│   ├── assets/
│   ├── saved_model.pb
│   └── variables/
├── model.pkl             # Modelo em pickle (backup)
├── history.pkl           # Histórico de treinamento
├── training_history.png  # Gráfico (se você plotar)
└── training_log.txt      # Log de execução (se você guardar)
```

---

## 7️⃣ Dicas de Performance

### ⚡ Transfer Learning (freeze_base=True)
```
✅ Converge rápido (10-20 epochs)
✅ Usa menos GPU
✅ Ideal para começar
❌ Performance pode ser limitada
```

### ⚡ Fine-tuning (freeze_base=False)
```
⚠️ Converge lento (20-50 epochs)
⚠️ Usa MUITA GPU
⚠️ Learning rate deve ser MUITO baixa
✅ Melhor performance final
✅ Use APÓS Transfer Learning convergir
```

### ⚡ Ensemble (múltiplos modelos)
```
Treinar 3 modelos com seeds diferentes:
model_final = (model1 + model2 + model3) / 3

✅ Reduz variância
✅ Predições mais robustas
❌ Treinamento 3x mais longo
```

---

## 8️⃣ Próximas Leituras

1. 📖 `TRANSFER_LEARNING_GUIDE.md` - Teoria detalhada
2. 📖 `RESNET_IMPLEMENTATION_SUMMARY.md` - Resumo visual
3. 📖 `models/resnet/training.py` - Código comentado

---

## 🎯 Checklist Final

- [ ] Instalar TensorFlow 2.8+
- [ ] Criar pasta `./result/resnet/`
- [ ] Executar Transfer Learning (freeze_base=True)
- [ ] Aguardar convergência (~30 epochs)
- [ ] Verificar resultados em `history.pkl`
- [ ] (Opcional) Executar Fine-tuning
- [ ] Comparar com tcc_II
- [ ] Documentar melhorias

---

**Bom treinamento! 🚀**

Se tiver dúvidas, revise o `TRANSFER_LEARNING_GUIDE.md` para entender melhor o funcionamento.
