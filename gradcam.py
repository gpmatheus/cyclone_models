"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation

Este script implementa Grad-CAM para visualizar as regiões de uma imagem que o modelo
utiliza para fazer previsões. Gera um novo dataset contendo mapas de calor sobrepostos
nas imagens originais.

Variáveis de ambiente:
    MODEL_PATH: Caminho para o modelo Keras (.h5 ou .keras)
    DATASET_PATH: Caminho para o dataset HDF5
    OUTPUT_PATH: (Opcional) Caminho para salvar o dataset com Grad-CAM
                 Padrão: 'data/gradcam_output.h5'
"""

import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model
import cv2
from typing import Tuple, Optional, Callable
import warnings

keras = tf.keras
Model = keras.models.Model

warnings.filterwarnings('ignore')

# ============================================================================
# Configurações de Variáveis de Ambiente
# ============================================================================
MODEL_PATH = os.getenv("MODEL_PATH") or "result/tcc_II/model.keras"
DATASET_PATH = os.getenv("DATASET_PATH") or "data/preprocessed/test.h5"
OUTPUT_PATH = os.getenv("OUTPUT_PATH") or "data/gradcam_output.h5"


class GradCAM:
    """
    Implementação de Grad-CAM para modelos Keras.
    
    Computa a importância de cada pixel da entrada para a previsão,
    gerando um mapa de calor que mostra quais regiões o modelo utiliza.
    """

    def __init__(self, model: keras.Model, layer_name: str):
        """
        Inicializa o GradCAM.

        Args:
            model: Modelo Keras treinado
            layer_name: Nome da camada convolucional para extrair os gradientes
        """
        self.model = model
        self.layer_name = layer_name
        
        # Encontrar a camada e criar um modelo intermediário
        # que retorna a saída da camada especificada
        layer = model.get_layer(layer_name)
        
        # Criar modelo intermediário que vai da entrada até a camada especificada
        self.intermediate_model = Model(
            inputs=model.inputs,
            outputs=layer.output
        )
        
        # Manter referência ao modelo original para predições
        self.prediction_model = model

    def compute_gradcam(
        self,
        input_image: np.ndarray,
        pred_index: Optional[int] = None,
    ) -> np.ndarray:
        """
        Computa o mapa de calor Grad-CAM para uma imagem.

        Args:
            input_image: Imagem de entrada (shape: (1, H, W, C))
            pred_index: Índice da classe. Se None, usa a classe predita.

        Returns:
            Mapa de calor normalizado (shape: (H, W))
        """
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Capturar outputs da camada intermediária
            conv_outputs = self.intermediate_model(input_image, training=True)
            tape.watch(conv_outputs)
            
            # Obter predições (usar modelo original)
            predictions = self.prediction_model(input_image, training=False)
            
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            else:
                pred_index = int(pred_index)
            
            class_channel = predictions[:, pred_index]

        # Computar gradientes da classe em relação aos outputs da camada
        grads = tape.gradient(class_channel, conv_outputs)

        # Computar os pesos (média dos gradientes)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Ponderar os mapas de ativação pelos pesos
        conv_outputs = conv_outputs[0]
        cam = conv_outputs @ pooled_grads[..., tf.newaxis]
        cam = tf.squeeze(cam, axis=-1)

        # Normalizar para 0-1
        cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-10)
        cam = tf.cast(cam * 255, tf.uint8)

        return cam.numpy()

    def compute_gradcam_batch(
        self,
        images: np.ndarray,
        class_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computa mapas de calor para um lote de imagens.

        Args:
            images: Lote de imagens (shape: (N, H, W, C))
            class_indices: Índices de classes para cada imagem

        Returns:
            Array de mapas de calor (shape: (N, H, W))
        """
        heatmaps = []
        
        for i, image in enumerate(images):
            image = np.expand_dims(image, axis=0)
            class_idx = class_indices[i] if class_indices is not None else None
            heatmap = self.compute_gradcam(image, class_idx)
            heatmaps.append(heatmap)
        
        return np.array(heatmaps)


def load_model_from_env() -> keras.Model:
    """
    Carrega o modelo usando a variável global MODEL_PATH.

    Returns:
        Modelo Keras carregado

    Raises:
        FileNotFoundError: Se o arquivo não existir
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
    
    print(f"Carregando modelo de: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    print(f"Modelo carregado com sucesso!")
    
    return model


def load_dataset_from_env() -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Carrega o dataset usando a variável global DATASET_PATH.
    
    Espera arquivo HDF5 com estrutura:
    - 'matrix': dataset com as imagens
    - 'info': tabela com as informações (ID, Vmax, time, etc.)

    Returns:
        Tupla (images, labels) carregada do HDF5

    Raises:
        FileNotFoundError: Se o arquivo não existir
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset não encontrado em: {DATASET_PATH}")
    
    print(f"Carregando dataset de: {DATASET_PATH}")
    
    # Carregar imagens
    with h5py.File(DATASET_PATH, 'r') as f:
        if 'matrix' not in f:
            raise KeyError(f"Chave 'matrix' não encontrada no arquivo. Chaves disponíveis: {list(f.keys())}")
        
        images = f['matrix'][:]
        print(f"Imagens carregadas: {images.shape}")
    
    # Carregar informações (labels)
    try:
        labels = pd.read_hdf(DATASET_PATH, key='info', mode='r')
        print(f"Informações carregadas: {labels.shape}")
    except (KeyError, FileNotFoundError):
        print("Aviso: Não foi possível carregar 'info' do arquivo")
        labels = None
    
    return images, labels


def crop_images_to_64x64(images: np.ndarray) -> np.ndarray:
    """
    Faz crop centralizado das imagens para 64x64.

    Args:
        images: Array de imagens (N, H, W, C)

    Returns:
        Array de imagens cropadas para 64x64
    """
    h, w = images.shape[1:3]
    target_size = 64
    
    if h == target_size and w == target_size:
        return images  # Já está no tamanho correto
    
    # Calcular o ponto de início para crop centralizado
    start_h = (h - target_size) // 2
    start_w = (w - target_size) // 2
    
    end_h = start_h + target_size
    end_w = start_w + target_size
    
    print(f"Cropando imagens de {images.shape[1:3]} para ({target_size}, {target_size})...")
    cropped = images[:, start_h:end_h, start_w:end_w, :]
    print(f"Imagens cropadas: {cropped.shape}")
    
    return cropped


def get_available_conv_layers(model: keras.Model) -> list:
    """
    Retorna lista de camadas convolucionais disponíveis no modelo.

    Args:
        model: Modelo Keras

    Returns:
        Lista com nomes das camadas convolucionais
    """
    conv_layers = []
    for layer in model.layers:
        if 'conv' in layer.name.lower():
            conv_layers.append(layer.name)
    return conv_layers


def main(
    layer_name: str = "conv2d_3",
    alpha: float = 0.4,
    batch_size: int = 32,
    output_path: Optional[str] = None,
) -> None:
    """
    Função principal que executa Grad-CAM no dataset.

    Args:
        layer_name: Nome da camada convolucional para Grad-CAM
                    (padrão: 'conv2d_3' - última camada convolucional)
        alpha: Transparência do mapa de calor (0-1)
        batch_size: Tamanho do lote para processamento
        output_path: Caminho para salvar o dataset com Grad-CAM.
                    Se None, usa variável global OUTPUT_PATH.

    Raises:
        FileNotFoundError: Se MODEL_PATH ou DATASET_PATH não forem encontrados
    """
    # Carregamento de configurações
    if output_path is None:
        output_path = OUTPUT_PATH
    
    print("=" * 80)
    print("GRAD-CAM - Visualização de Ativações")
    print("=" * 80)
    
    # Carregar modelo e dataset
    model = load_model_from_env()
    images, labels = load_dataset_from_env()
    
    # Fazer crop para 64x64
    images = crop_images_to_64x64(images)
    
    # Inicializar Grad-CAM
    gradcam = GradCAM(model, layer_name)
    
    # Predições para obter os índices de classe
    print(f"\nGerando previsões para {len(images)} imagens...")
    predictions = model.predict(images, batch_size=batch_size, verbose=1)
    class_indices = np.argmax(predictions, axis=1)
    
    # Processar em lotes
    print(f"\nGerando mapas de calor Grad-CAM...")
    gradcam_images = []
    
    num_batches = int(np.ceil(len(images) / batch_size))
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(images))
        
        batch_images = images[start_idx:end_idx]
        batch_classes = class_indices[start_idx:end_idx]
        
        # Computar Grad-CAM
        heatmaps = gradcam.compute_gradcam_batch(batch_images, batch_classes)
        
        gradcam_images.extend(heatmaps)
        
        print(f"Lote {batch_idx + 1}/{num_batches} processado")
    
    gradcam_images = np.array(gradcam_images)
    
    # Salvar dataset com Grad-CAM
    print(f"\nSalvando dataset em: {output_path}")
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('matrix', data=gradcam_images)
        f.create_dataset('class_predictions', data=class_indices)
        
        # Salvar metadados
        f.attrs['layer_name'] = layer_name
        f.attrs['alpha'] = alpha
        f.attrs['num_samples'] = len(gradcam_images)
    
    # Salvar informações como tabela do pandas se disponível
    if labels is not None:
        try:
            labels.to_hdf(output_path, key='info', mode='a', format='table')
        except Exception as e:
            print(f"Aviso: Não foi possível salvar 'info' com pandas: {e}")
    
    print(f"Dataset salvo com sucesso!")
    print(f"Forma do dataset: {gradcam_images.shape}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Gera mapas de calor Grad-CAM para um dataset usando um modelo Keras"
    )
    parser.add_argument(
        "--layer-name",
        type=str,
        default="conv2d_3",
        help="Nome da camada convolucional para Grad-CAM (padrão: conv2d_3)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Transparência do mapa de calor (0-1, padrão: 0.4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do lote (padrão: 32)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Caminho para salvar o dataset. Padrão: variável OUTPUT_PATH ou 'data/gradcam_output.h5'"
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Lista todas as camadas convolucionais disponíveis no modelo e sai"
    )
    
    args = parser.parse_args()
    
    # Se --list-layers foi especificado, apenas listar camadas e sair
    if args.list_layers:
        try:
            model = load_model_from_env()
            conv_layers = get_available_conv_layers(model)
            print("=" * 80)
            print("CAMADAS CONVOLUCIONAIS DISPONÍVEIS")
            print("=" * 80)
            for i, layer_name in enumerate(conv_layers, 1):
                print(f"{i}. {layer_name}")
            print("=" * 80)
            print(f"\nUse: python gradcam.py --layer-name <nome_da_camada>")
        except Exception as e:
            print(f"Erro ao listar camadas: {e}")
        exit(0)
    
    try:
        main(
            layer_name=args.layer_name,
            alpha=args.alpha,
            batch_size=args.batch_size,
            output_path=args.output,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Erro: {e}")
        exit(1)
    except Exception as e:
        print(f"Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
