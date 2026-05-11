import tensorflow as tf
import numpy as np
from pathlib import Path
import pickle
import os
import random

keras = tf.keras

try:
    from ... import data
except ImportError:
    import data

result_path_folder = os.getenv("RESULT_PATH") or ""

is_remote = os.environ.get("KAGGLE_URL_BASE") is not None


def parse_example(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_image_tf(image)
    return image, label


def preprocess_image_tf(image):
    """
    Pré-processa a imagem com rotação aleatória.
    ResNet50 espera imagens RGB (3 canais), então adaptamos o shape se necessário.
    """
    angle_rad = tf.random.uniform([], 0, 2 * np.pi)
    image_shape = tf.shape(image)[0:2]
    cx = tf.cast(image_shape[1] / 2, tf.float32)
    cy = tf.cast(image_shape[0] / 2, tf.float32)
    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)
    transform = tf.stack(
        [
            cos_a,
            -sin_a,
            (1 - cos_a) * cx + sin_a * cy,
            sin_a,
            cos_a,
            (1 - cos_a) * cy - sin_a * cx,
            0.0,
            0.0,
        ]
    )
    transform = tf.reshape(transform, [8])
    transform = tf.expand_dims(transform, 0)
    image = tf.expand_dims(image, 0)
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transform,
        output_shape=image_shape,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    rotated = tf.squeeze(rotated, 0)
    # Redimensionar para 224x224 (tamanho padrão de entrada da ResNet50)
    resized = tf.image.resize_with_crop_or_pad(rotated, 224, 224)
    
    # Se houver apenas 1 canal, replicar para 3 canais (RGB)
    if tf.shape(resized)[-1] == 1:
        resized = tf.repeat(resized, 3, axis=-1)
    # Se houver 2 canais, adicionar um terceiro
    elif tf.shape(resized)[-1] == 2:
        zero_channel = tf.zeros_like(resized[:, :, :1])
        resized = tf.concat([resized, zero_channel], axis=-1)
    
    return resized


def build_dataset(data, batch, seed=None, sample_pct=1.0):

    reproduc = seed is not None

    imgs, labels = data

    imgs = imgs.astype("float32")
    labels = labels["Vmax"].astype("float32")

    images_len = imgs.shape[0]
    print(f"Dataset len: {images_len}")

    images_sample_len = int(images_len * sample_pct)
    print(f"Dataset new len: {images_sample_len}")

    idx = list(range(images_len))
    random.shuffle(idx)
    idx = idx[:images_sample_len]

    imgs = imgs[idx]
    labels = labels.iloc[idx].reset_index(drop=True)

    dataset = tf.data.Dataset.from_tensor_slices((imgs, labels))
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=len(imgs), seed=seed)
    
    # Adiciona opções determinísticas se reprodutibilidade for necessária
    if reproduc:
        options = tf.data.Options()
        options.experimental_deterministic = True
        dataset = dataset.with_options(options)
    
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset, imgs.shape


def load_datasets(channels, generated_channels, img_w, batch, sample_pct, seed=None, force=True):
    train, valid, _ = data.preprocess(channels, generated_channels, img_w, force=force)
    train_ds = build_dataset(train, batch, seed=seed, sample_pct=sample_pct)
    valid_ds = build_dataset(valid, batch, seed=seed)
    return train_ds, valid_ds


def build_model_with_resnet(input_shape, lr, l2_regularizer=1e-5, freeze_base=True):
    """
    Constrói um modelo usando ResNet50 pré-treinada com transfer learning.
    
    Parâmetros:
    -----------
    input_shape : tuple
        Shape da imagem de entrada (224, 224, 3) para ResNet50
    
    lr : float
        Taxa de aprendizado
    
    l2_regularizer : float
        Fator de regularização L2 para evitar overfitting
    
    freeze_base : bool
        Se True, congela os pesos da ResNet50 pré-treinada (transfer learning)
        Se False, descongelaria todos os pesos para fine-tuning
    
    Retorna:
    --------
    model : keras.Model
        Modelo compilado pronto para treinamento
    
    ============================================================================
    EXPLICAÇÃO DO FUNCIONAMENTO:
    ============================================================================
    
    1. CARREGAMENTO DA RESNET152 PRÉ-TREINADA:
       - ResNet50 foi treinada no dataset ImageNet com ~1.2 milhões de imagens
       - Já aprendeu a extrair features visuais (bordas, texturas, padrões complexos)
       - Vem com 152 camadas convolucionais que criaram representações robustas
    
    2. REMOÇÃO DO "HEAD" DE CLASSIFICAÇÃO:
       - A ResNet50 original tem 1000 saídas (classes ImageNet)
       - Removemos a última camada (include_top=False)
       - Ficamos com a base convolucional que extrai features
    
    3. CONGELAMENTO DOS PESOS (TRANSFER LEARNING):
       - Se freeze_base=True, os pesos pré-treinados NÃO são atualizados
       - Apenas o "head" customizado (nossas camadas Dense) aprende
       - Vantagem: treinamento muito mais rápido (poucos parâmetros)
       - Mantém o conhecimento de features do ImageNet
    
    4. ADIÇÃO DE CAMADAS CUSTOMIZADAS (HEAD):
       - GlobalAveragePooling2D: reduz o output da ResNet de (7,7,2048) para (2048,)
       - Dense(512) com ReLU: processa as features extraídas
       - Dropout(0.5): regularização para evitar overfitting
       - Dense(64) com ReLU: reduz ainda mais a dimensionalidade
       - Dense(1) Linear: prediz um único valor (velocidade do ciclone)
    
    5. POR QUE ISSO FUNCIONA PARA CICLONES:
       - ResNet aprendeu a reconhecer padrões visuais genéricos
       - Ciclones têm padrões visuais: forma espiral, olho central, etc
       - A ResNet pode identificar essas estruturas
       - Apenas precisamos adaptar para regressão (velocidade) em vez de classificação
    
    ============================================================================
    """
    
    # ========== CARREGA RESNET50 PRÉ-TREINADA ==========
    # weights='imagenet': carrega pesos do ImageNet
    # include_top=False: remove as 1000 camadas de classificação final
    # input_shape=(224, 224, 3): tamanho padrão de entrada
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling=None,
    )
    
    # ========== CONGELA PESOS DA BASE (TRANSFER LEARNING) ==========
    # Se freeze_base=True, não atualiza os pesos da ResNet50 durante treinamento
    # Apenas as novas camadas que adicionaremos aprenderão
    base_model.trainable = freeze_base is False
    
    # Se decidirmos fazer fine-tuning, podemos congelar apenas as primeiras camadas
    if freeze_base:
        print("✓ Pesos da ResNet50 congelados (Transfer Learning)")
    else:
        print("✓ Pesos da ResNet50 descongelados (Fine-tuning completo)")
    
    # ========== CONSTRÓI MODELO SEQUENCIAL COM RESNET COMO BASE ==========
    model = keras.models.Sequential([
        keras.layers.Input(shape=(224, 224, 3)),
        
        # === BASE: ResNet50 pré-treinada ===
        # Extrai features hierárquicas das imagens
        base_model,
        
        # === CAMADAS CUSTOMIZADAS PARA REGRESSÃO ===
        
        # Flatten: converte (7, 7, 2048) -> (100352,)
        keras.layers.Flatten(),
        
        # Dense com 512 neurônios: processa as features extraídas
        keras.layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(l2_regularizer),
            bias_regularizer=keras.regularizers.L2(l2_regularizer),
        ),
        
        # Dense com 64 neurônios: reduz a dimensionalidade
        keras.layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(l2_regularizer),
            bias_regularizer=keras.regularizers.L2(l2_regularizer),
        ),
        
        # === OUTPUT: Regressão ===
        # Dense com 1 neurônio: prediz um único valor (velocidade Vmax)
        # activation='linear': sem ativação, permite qualquer valor real
        keras.layers.Dense(
            1,
            activation='linear',
            kernel_regularizer=keras.regularizers.L2(l2_regularizer),
        ),
    ])
    
    # ========== COMPILAÇÃO DO MODELO ==========
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mse',  # Mean Squared Error para regressão
        metrics=['mse', 'mae', keras.metrics.RootMeanSquaredError()],
    )
    
    return model


def train_model(
    model,
    train_ds,
    valid_ds,
    epochs,
    batch,
    patience,
):
    train_ds, train_shape = train_ds
    valid_ds, valid_shape = valid_ds

    steps_per_epoch = train_shape[0] // batch
    validation_steps = valid_shape[0] // batch

    # Para o treinamento quando a val_loss não melhora por `patience` épocas
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    # Reduz o LR quando a val_loss estagna (complementa o CosineDecay)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=patience // 2,
        min_lr=1e-7,
        verbose=1,
    )

    with tf.device("/GPU:0"):
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=[early_stopping, reduce_lr],
        )

    return model, history


def save_model(model, path):

    model.export(Path(f"{path}/model"))
    model.save(f"{path}/model.keras")

    with open(f"{path}/model.pkl", "wb") as file:
        pickle.dump(model, file)

def save_history(history, path):
    with open(f"{path}/history.pkl", "wb") as file:
        pickle.dump(history.history, file)


def set_seed(seed):
    """Configure todas as seeds para reproducibilidade"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Força TensorFlow a ser determinístico (mais lento, mas reproduzível)
    tf.config.experimental.enable_op_determinism()
    
    # Desativa otimizações não-determinísticas
    tf.config.run_functions_eagerly(True)



def main(
    channels=[0, 3],
    generated_channels=[0],
    img_w=64,
    batch=8,
    learning_rate=5e-5,
    epochs=500,
    sample_pct=1.0,
    seed=None,
    patience=30,
    l2_regularizer=1e-5,
    force=True,
    freeze_base=True,
):
    """
    Executar treinamento da ResNet50 com Transfer Learning
    
    Parâmetros:
    -----------
    channels : list, default=[0, 3]
        Canais de entrada
    generated_channels : list, default=[0]
        Canais gerados
    img_w : int, default=64
        Largura da imagem
    batch : int, default=8
        Batch size
    learning_rate : float, default=5e-5
        Learning rate
    epochs : int, default=500
        Número de epochs
    sample_pct : float, default=1.0
        Percentual de dados a usar
    seed : int, default=None
        Seed para reproducibilidade
    patience : int, default=30
        Patience para early stopping
    l2_regularizer : float, default=1e-5
        Fator de regularização L2
    force : bool, default=True
        Reprocessar dados
    freeze_base : bool, default=True
        Congelar pesos da ResNet50 (Transfer Learning)
    """
    
    if seed is not None: 
        set_seed(seed)

    train_ds, valid_ds = load_datasets(channels, generated_channels, img_w, batch, sample_pct, seed=seed, force=force)

    # Constrói modelo com ResNet50
    model = build_model_with_resnet(
        (224, 224, 3),  # ResNet50 sempre usa 224x224x3
        learning_rate,
        l2_regularizer,
        freeze_base=freeze_base
    )
    model.summary()
    model, history = train_model(model, train_ds, valid_ds, epochs, batch, patience)

    save_model(model, result_path_folder)
    save_history(history, result_path_folder)


if __name__ == "__main__":
    main()
