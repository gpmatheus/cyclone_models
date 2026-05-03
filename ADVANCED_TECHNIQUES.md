# 🔬 Exemplos Avançados: Fine-tuning Progressivo, Ensemble e Técnicas Customizadas

## 1️⃣ Fine-tuning Progressivo (Técnica Recomendada)

Esta é a técnica mais eficaz para obter máxima performance. Descongelamos camadas gradualmente.

### Estratégia: Descongelar Progressivamente

```python
#!/usr/bin/env python3
"""
Fine-tuning Progressivo - Descongelamos camadas gradualmente
Etapa 1: Transfer Learning (ResNet congelada)
Etapa 2: Fine-tuning parcial (últimas camadas convolução)
Etapa 3: Fine-tuning completo (rede inteira)
"""

import os
import sys
import tensorflow as tf

os.environ['RESULT_PATH'] = './result/resnet'
os.environ['DATA_PATH'] = './data'
os.environ['PREPROCESSED_PATH'] = './data/preprocessed'

sys.path.insert(0, './models')

from resnet import training

# ============================================================================
# ETAPA 1: TRANSFER LEARNING (ResNet congelada, apenas head aprende)
# ============================================================================

print("\n" + "="*60)
print("ETAPA 1: TRANSFER LEARNING")
print("="*60)
print("Treinando apenas o head customizado...")

model1, history1 = training.main(
    channels=[0, 3],
    batch=8,
    learning_rate=5e-5,      # Taxa normal
    epochs=100,
    seed=42,
    patience=30,
    freeze_base=True         # ❄️ ResNet congelada
)

# Analisar resultados
val_loss_1 = min(history1.history['val_loss'])
val_mae_1 = min(history1.history['val_mae'])
best_epoch_1 = history1.history['val_loss'].index(val_loss_1) + 1

print(f"\n✅ ETAPA 1 Completa!")
print(f"   Melhor epoch: {best_epoch_1}")
print(f"   Val Loss: {val_loss_1:.4f}")
print(f"   Val MAE: {val_mae_1:.2f} nós")

# Se convergiu bem, fazer fine-tuning
if val_mae_1 < 15:  # MAE razoável
    print("\n✅ Performance aceitável! Passando para Etapa 2 (Fine-tuning)...")
else:
    print("\n⚠️ Performance ainda pode melhorar. Execute Etapa 1 novamente com:")
    print("    learning_rate=1e-4 ou epochs=150")
    sys.exit()

# ============================================================================
# ETAPA 2: FINE-TUNING PARCIAL (Descongelar últimas 2 camadas convolucionais)
# ============================================================================

print("\n" + "="*60)
print("ETAPA 2: FINE-TUNING PARCIAL")
print("="*60)
print("Descongelando últimas 2 blocos da ResNet (Blocos 5 e 4)...")

# Carregar modelo da Etapa 1
model2 = tf.keras.models.load_model('./result/resnet/model.keras')

# Acessar a base (ResNet152)
base_model = model2.layers[1]  # Segunda camada é a ResNet152

# Descongelar APENAS os últimos blocos
# ResNet152 tem 5 blocos: Bloco 5 é as últimas 3 camadas (índices ~150-152)
# Bloco 4 é as 36 camadas anteriores (índices ~114-149)
# Vamos descongelar a partir da camada 100

print(f"\nTotal de camadas na ResNet152: {len(base_model.layers)}")

# Descongelar camadas a partir do índice 100
for layer in base_model.layers[:100]:
    layer.trainable = False  # Manter congeladas

for layer in base_model.layers[100:]:
    layer.trainable = True   # Descongelar últimas camadas

print(f"✓ Camadas congeladas: 0-99 (100 camadas)")
print(f"✓ Camadas descongeladas: 100-{len(base_model.layers)-1} ({len(base_model.layers)-100} camadas)")

# Recompilar com learning rate MUITO menor
model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-7),  # ⚠️ MUITO BAIXO
    loss='mse',
    metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()],
)

print("\nRecompilado com learning_rate=1e-7 (mil vezes menor que Etapa 1)")

# Treinar mais
from models.resnet.training import load_datasets, train_model

train_ds, valid_ds = load_datasets(
    channels=[0, 3],
    generated_channels=[0],
    img_w=64,
    batch=4,  # Batch menor (mais parâmetros treináveis)
    sample_pct=1.0,
    seed=42,
    force=False
)

print("\nTreinando (Etapa 2)...")
model2, history2 = train_model(
    model2,
    train_ds,
    valid_ds,
    epochs=30,      # Menos epochs (já treinou bastante)
    batch=4,
    patience=15
)

val_mae_2 = min(history2.history['val_mae'])
print(f"\n✅ ETAPA 2 Completa!")
print(f"   Val MAE: {val_mae_2:.2f} nós")
print(f"   Melhora: {val_mae_1 - val_mae_2:.2f} nós ({((val_mae_1 - val_mae_2)/val_mae_1)*100:.1f}%)")

# ============================================================================
# ETAPA 3: FINE-TUNING COMPLETO (Descongelar TUDO)
# ============================================================================

if val_mae_2 < val_mae_1 - 0.5:  # Se houve melhora >0.5 nós
    print("\n" + "="*60)
    print("ETAPA 3: FINE-TUNING COMPLETO")
    print("="*60)
    print("Descongelando TODA a ResNet152...")

    # Descongelar base inteira
    for layer in base_model.layers:
        layer.trainable = True

    print(f"✓ Todas as {len(base_model.layers)} camadas da ResNet agora treináveis")

    # Recompilar com learning rate ainda MAIS baixa
    model2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-8),  # 10 vezes menor ainda
        loss='mse',
        metrics=['mse', 'mae', tf.keras.metrics.RootMeanSquaredError()],
    )

    print("Recompilado com learning_rate=1e-8")
    print("\nTreinando (Etapa 3)...")

    model2, history3 = train_model(
        model2,
        train_ds,
        valid_ds,
        epochs=20,
        batch=4,
        patience=10
    )

    val_mae_3 = min(history3.history['val_mae'])
    print(f"\n✅ ETAPA 3 Completa!")
    print(f"   Val MAE: {val_mae_3:.2f} nós")
    print(f"   Melhora total: {val_mae_1 - val_mae_3:.2f} nós ({((val_mae_1 - val_mae_3)/val_mae_1)*100:.1f}%)")

    # Salvar modelo final
    from models.resnet.training import save_model
    save_model(model2, './result/resnet_finetuned')
    print(f"\n✅ Modelo fine-tuned salvo em: ./result/resnet_finetuned/")

else:
    print("\n⚠️ Etapa 2 não melhorou o suficiente. Não fazer Etapa 3.")
    print("   Considere aumentar epochs da Etapa 2 ou ajustar learning rate.")

# ============================================================================
# RESUMO FINAL
# ============================================================================

print("\n" + "="*60)
print("RESUMO DO FINE-TUNING PROGRESSIVO")
print("="*60)
print(f"\nEtapa 1 (Transfer Learning):")
print(f"  MAE: {val_mae_1:.2f} nós")
print(f"  Camadas congeladas: Todas as 152 da ResNet")

print(f"\nEtapa 2 (Fine-tuning Parcial):")
print(f"  MAE: {val_mae_2:.2f} nós")
print(f"  Melhora: {val_mae_1 - val_mae_2:.2f} nós")
print(f"  Camadas descongeladas: Últimas 52 da ResNet")

if val_mae_2 < val_mae_1 - 0.5:
    print(f"\nEtapa 3 (Fine-tuning Completo):")
    print(f"  MAE: {val_mae_3:.2f} nós")
    print(f"  Melhora: {val_mae_1 - val_mae_3:.2f} nós")
    print(f"  Camadas descongeladas: TODAS as 152 da ResNet")
```

---

## 2️⃣ Ensemble: Múltiplos Modelos

Treinar múltiplos modelos e fazer média reduz variância e melhora robustez.

```python
#!/usr/bin/env python3
"""
Ensemble de Modelos: treina 3 ResNet152 com seeds diferentes
Predição final é a média das 3
"""

import os
import sys
import tensorflow as tf
import numpy as np

os.environ['RESULT_PATH'] = './result/resnet'
os.environ['DATA_PATH'] = './data'
os.environ['PREPROCESSED_PATH'] = './data/preprocessed'

sys.path.insert(0, './models')

from resnet import training

# ============================================================================
# TREINAR 3 MODELOS COM SEEDS DIFERENTES
# ============================================================================

print("\n" + "="*60)
print("CRIANDO ENSEMBLE DE 3 MODELOS")
print("="*60)

models = []
seeds = [42, 123, 456]
histories = []

for i, seed in enumerate(seeds, 1):
    print(f"\n📊 Treinando Modelo {i}/3 (seed={seed})...")
    
    model, history = training.main(
        channels=[0, 3],
        batch=8,
        learning_rate=5e-5,
        epochs=50,
        seed=seed,
        patience=30,
        freeze_base=True
    )
    
    models.append(model)
    histories.append(history)
    
    val_mae = min(history.history['val_mae'])
    print(f"✅ Modelo {i} finalizado - Val MAE: {val_mae:.2f} nós")

print("\n✅ Todos os 3 modelos treinados!")

# ============================================================================
# FUNÇÃO PARA PREDIÇÃO COM ENSEMBLE
# ============================================================================

def predict_with_ensemble(image, models):
    """
    Predizer usando ensemble (média de múltiplos modelos)
    
    Args:
        image: numpy array (H, W, C)
        models: lista de modelos TensorFlow
    
    Returns:
        vmax_pred: valor predito
        vmax_std: desvio padrão (indicador de incerteza)
    """
    
    # Pré-processar imagem
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    
    # Garantir 3 canais
    if image.shape[-1] == 1:
        image = tf.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 2:
        zero_channel = tf.zeros_like(image[:, :, :1])
        image = tf.concat([image, zero_channel], axis=-1)
    
    # Batch
    image = tf.expand_dims(image, 0)
    
    # Predições de cada modelo
    predictions = []
    for model in models:
        pred = model.predict(image, verbose=0)
        predictions.append(float(pred[0][0]))
    
    # Média e desvio padrão
    vmax_pred = np.mean(predictions)
    vmax_std = np.std(predictions)
    
    return vmax_pred, vmax_std, predictions


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

print("\n" + "="*60)
print("EXEMPLOS DE PREDIÇÃO COM ENSEMBLE")
print("="*60)

# Criar imagem fake para teste
test_image = np.random.randn(64, 64, 2).astype(np.float32)

vmax_pred, vmax_std, individual_preds = predict_with_ensemble(test_image, models)

print(f"\n📊 Predições individuais:")
for i, pred in enumerate(individual_preds, 1):
    print(f"   Modelo {i}: {pred:.2f} nós")

print(f"\n📊 Predição do Ensemble:")
print(f"   Média:    {vmax_pred:.2f} nós")
print(f"   Desvio:   ±{vmax_std:.2f} nós")
print(f"   Intervalo: [{vmax_pred-vmax_std:.2f}, {vmax_pred+vmax_std:.2f}] nós")

if vmax_std < 2:
    print(f"   ✅ Confiança: ALTA (baixa variância entre modelos)")
elif vmax_std < 5:
    print(f"   ⚠️  Confiança: MÉDIA")
else:
    print(f"   ❌ Confiança: BAIXA (alta variância entre modelos)")

# ============================================================================
# SALVAR ENSEMBLE PARA REUTILIZAÇÃO
# ============================================================================

import pickle

ensemble_data = {
    'models': models,
    'seeds': seeds,
    'histories': histories
}

with open('./result/resnet_ensemble.pkl', 'wb') as f:
    pickle.dump(ensemble_data, f)

print(f"\n✅ Ensemble salvo em: ./result/resnet_ensemble.pkl")
```

---

## 3️⃣ Data Augmentation Customizada

Aumentar dados sinteticamente para melhorar robustez.

```python
#!/usr/bin/env python3
"""
Data Augmentation Customizada para imagens de ciclones
"""

import tensorflow as tf
import numpy as np

# ============================================================================
# AUGMENTATION PIPELINE CUSTOMIZADO
# ============================================================================

class CycloneAugmentation:
    """Camadas de augmentation específicas para ciclones tropicais"""
    
    def __init__(self):
        """Inicializar camadas de augmentation"""
        
        self.augmentation = tf.keras.Sequential([
            # Rotação aleatória: simula ciclone em diferentes ângulos
            tf.keras.layers.RandomRotation(0.5),
            
            # Zoom aleatório: simula aproximação/afastamento
            tf.keras.layers.RandomZoom(0.2),
            
            # Flip horizontal: simetria do ciclone
            tf.keras.layers.RandomFlip("horizontal"),
            
            # Contrast: simula variação de intensidade radiativa
            tf.keras.layers.RandomContrast(0.2),
            
            # Brightness: simula variação de temperatura
            tf.keras.layers.RandomBrightness(0.1),
            
            # Gaussian noise: simula ruído de sensor
            tf.keras.layers.GaussianNoise(0.05),
        ])
    
    def augment(self, image):
        """Aplicar augmentation a uma imagem"""
        return self.augmentation(image, training=True)
    
    def augment_batch(self, images):
        """Aplicar augmentation a um batch"""
        return tf.map_fn(
            self.augment,
            images,
            fn_output_signature=images.dtype
        )


# ============================================================================
# USAR COM DATASET
# ============================================================================

def create_augmented_dataset(images, labels, batch_size=8):
    """Criar dataset com augmentation"""
    
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size)
    
    augmenter = CycloneAugmentation()
    
    def augment_batch(imgs, lbls):
        imgs = augmenter.augment_batch(imgs)
        return imgs, lbls
    
    dataset = dataset.map(
        augment_batch,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    return dataset


# ============================================================================
# VISUALIZAR AUGMENTATIONS
# ============================================================================

if __name__ == "__main__":
    # Criar imagem fake
    test_image = np.random.randn(224, 224, 3).astype(np.float32)
    test_image = np.expand_dims(test_image, 0)
    
    augmenter = CycloneAugmentation()
    
    print("Gerando 5 versões augmentadas da mesma imagem...")
    for i in range(5):
        augmented = augmenter.augment_batch(test_image)
        print(f"Versão {i+1}: {augmented.shape}")
    
    print("✅ Augmentations funcionando corretamente!")
```

---

## 4️⃣ Validação Cruzada (K-Fold)

Usar k-fold cross-validation para validação mais robusta.

```python
#!/usr/bin/env python3
"""
K-Fold Cross-Validation com ResNet152
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

os.environ['RESULT_PATH'] = './result/resnet'
os.environ['DATA_PATH'] = './data'
os.environ['PREPROCESSED_PATH'] = './data/preprocessed'

sys.path.insert(0, './models')

from resnet import training

# ============================================================================
# K-FOLD CROSS-VALIDATION
# ============================================================================

def kfold_cross_validation(images, labels, k=5, epochs=50):
    """
    K-Fold Cross-Validation
    
    Args:
        images: array de imagens
        labels: array de labels
        k: número de folds
        epochs: epochs por fold
    
    Returns:
        scores: lista de MAE para cada fold
    """
    
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    
    fold_num = 1
    for train_idx, val_idx in kfold.split(images):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num}/{k}")
        print(f"{'='*60}")
        
        # Split
        x_train, x_val = images[train_idx], images[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Treinar modelo
        model = training.build_model_with_resnet(
            (224, 224, 3),
            lr=5e-5,
            freeze_base=True
        )
        
        # ... treinar ...
        # mae = avaliar(model, x_val, y_val)
        
        # scores.append(mae)
        print(f"✅ Fold {fold_num} finalizado")
        
        fold_num += 1
    
    print(f"\n{'='*60}")
    print("RESUMO K-FOLD")
    print(f"{'='*60}")
    print(f"MAE por fold: {scores}")
    print(f"MAE média: {np.mean(scores):.2f} ± {np.std(scores):.2f} nós")
    
    return scores


# ============================================================================
# USAR
# ============================================================================

# images = carregar_imagens()
# labels = carregar_labels()
# scores = kfold_cross_validation(images, labels, k=5)
```

---

## 📚 Quando usar cada técnica?

```
┌─────────────────────────────────────────────────────┐
│ TÉCNICA             QUANDO USAR      GANHO ESPERADO │
├─────────────────────────────────────────────────────┤
│ Transfer Learning   Começar         ~10-15 kt MAE  │
│                     (padrão)                        │
│                                                     │
│ Fine-tuning Prog.   Performance     +5-10% melhora │
│                     já boa, quer    (~2-3 kt MAE) │
│                     refinar                        │
│                                                     │
│ Ensemble (3-5)      Performance     +10-20% melhora│
│                     ótima, quer     (mais robusto)│
│                     robustez                       │
│                                                     │
│ Data Augmentation   Poucos dados    +5-10% melhora│
│                     (<500 imagens)                 │
│                                                     │
│ K-Fold Validation   Publicação/     Confiabilidade│
│                     Production      aumentada      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

✨ **Parabéns! Você agora tem técnicas avançadas para otimizar seu modelo!** ✨

