"""
Script de Análise de Erro do Modelo
Carrega um modelo treinado e calcula erros de predição no conjunto de testes.
Gera gráficos de distribuição de frequência de erros absolutos.

Variáveis de ambiente:
  MODEL_PATH: Caminho para o arquivo do modelo treinado (.keras ou .h5)
  DATASET_PATH: Caminho para o conjunto de testes (arquivo HDF5)
  OUTPUT_PATH: Diretório para salvar resultados
"""

import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# Environment variables
MODEL_PATH = os.getenv('MODEL_PATH') or 'result/tcc_II/2/model.keras'
DATASET_PATH = os.getenv('DATASET_PATH') or 'data/preprocessed/test.h5'
OUTPUT_PATH = os.getenv('OUTPUT_PATH') or 'result/plot'

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    # from tensorflow import keras
    keras = tf.keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠️  TensorFlow/Keras não disponível. Instale com: pip install tensorflow")


def load_model(model_path):
    """Carrega o modelo treinado."""
    print(f"Carregando modelo de {model_path}...")
    model = keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer='adam',
        loss=keras.losses.MeanSquaredError(),
        metrics=['mse']
    )
    print("Modelo carregado com sucesso!")
    return model

def get_images_slice(images_shape, width):
    # Calcula o slice para cortar a imagem centralizada com a largura especificada
    start = images_shape[1] // 2 - width // 2
    end = images_shape[1] // 2 + width // 2
    return slice(start, end)

def cut_images(images, width):
    # Corta as imagens para o tamanho especificado, centralizando o corte
    slc = get_images_slice(images.shape, width)
    return images[:, slc, slc, :]

def load_test_dataset(dataset_path, crop_w):
    """Carrega dados de teste do arquivo HDF5."""
    print(f"Carregando dados de teste de {dataset_path}...")
    with h5py.File(dataset_path, mode='r') as file:
        images = file['matrix'][:]
        images = tf.image.resize_with_crop_or_pad(images, crop_w, crop_w)
        
    
    info = pd.read_hdf(dataset_path, key="info", mode="r")["Vmax"]
    print(f"Dados carregados! Total de amostras: {len(images)}")

    return images, info


def calculate_errors(model, X, y):
    """Calcula erros de predição para cada amostra."""
    print(f"\nFazendo predições em {len(X)} amostras...")
    
    try:
        # Make predictions
        predictions = model.predict(X, verbose=0)
        
        # Handle multi-dimensional predictions (take first output if needed)
        if len(predictions.shape) > 1:
            predictions = predictions[:, 0]
        
        print(f"✓ Predições concluídas")
        print(f"  Forma das predições: {predictions.shape}")
        print(f"  Intervalo das predições: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
        
        if y is not None:
            # Calculate raw errors
            raw_errors = y - predictions
            
            # Calculate absolute errors
            abs_errors = np.abs(raw_errors)
            
            print(f"\n✓ Erros calculados")
            print(f"  Erros brutos - média: {np.mean(raw_errors):.4f}, desvio: {np.std(raw_errors):.4f}")
            print(f"  Erros absolutos - média: {np.mean(abs_errors):.4f}, desvio: {np.std(abs_errors):.4f}")
            print(f"  Erros absolutos - mín: {np.min(abs_errors):.4f}, máx: {np.max(abs_errors):.4f}")
            
            return predictions, raw_errors, abs_errors
        else:
            return predictions, None, None
            
    except Exception as e:
        raise RuntimeError(f"Falha ao fazer predições: {e}")


def create_error_distribution_plot(abs_errors, output_path):
    """Cria gráfico de distribuição de frequência de erros absolutos."""
    print(f"\nGerando gráfico de distribuição de frequência...")
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histograma
    n, bins, patches = ax.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Erro Absoluto', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequência', fontsize=11, fontweight='bold')
    ax.set_title('Distribuição de Frequência de Erros Absolutos de Predição', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Adicionar linha da média
    mean_val = np.mean(abs_errors)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, label=f'Média: {mean_val:.4f}')
    ax.legend(loc='upper left', fontsize=10)
    
    # Adicionar caixa de estatísticas
    stats_text = (
        f"Média: {np.mean(abs_errors):.4f}\n"
        f"Mediana: {np.median(abs_errors):.4f}\n"
        f"Desvio Padrão: {np.std(abs_errors):.4f}\n"
        f"Mínimo: {np.min(abs_errors):.4f}\n"
        f"Máximo: {np.max(abs_errors):.4f}\n"
        f"Total: {len(abs_errors)}"
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontfamily='monospace', fontsize=9)
    
    plt.tight_layout()
    
    # Salvar figura
    output_file = Path(output_path) / "error_distribution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Gráfico salvo em {output_file}")
    plt.close()


def save_error_statistics(abs_errors, raw_errors, output_path):
    """Salva estatísticas de erros como JSON."""
    stats = {
        "absolute_errors": {
            "mean": float(np.mean(abs_errors)),
            "median": float(np.median(abs_errors)),
            "std": float(np.std(abs_errors)),
            "min": float(np.min(abs_errors)),
            "max": float(np.max(abs_errors)),
            "count": int(len(abs_errors))
        },
        "raw_errors": {
            "mean": float(np.mean(raw_errors)),
            "median": float(np.median(raw_errors)),
            "std": float(np.std(raw_errors)),
            "min": float(np.min(raw_errors)),
            "max": float(np.max(raw_errors)),
        },
        "percentiles": {
            "25th": float(np.percentile(abs_errors, 25)),
            "50th": float(np.percentile(abs_errors, 50)),
            "75th": float(np.percentile(abs_errors, 75)),
            "90th": float(np.percentile(abs_errors, 90)),
            "95th": float(np.percentile(abs_errors, 95)),
            "99th": float(np.percentile(abs_errors, 99)),
        },
        "timestamp": datetime.now().isoformat()
    }
    
    output_file = Path(output_path) / "error_statistics.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Estatísticas salvas em {output_file}")


def save_error_csv(abs_errors, raw_errors, output_path):
    """Salva erros em arquivo CSV para análise adicional."""
    import csv
    
    output_file = Path(output_path) / "errors.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID_Amostra', 'Erro_Bruto', 'Erro_Absoluto'])
        for idx, (raw_err, abs_err) in enumerate(zip(raw_errors, abs_errors)):
            writer.writerow([idx, raw_err, abs_err])
    
    print(f"✓ Erros salvos em {output_file}")


def main(crop_w=64):
    """Execução principal."""
    try:
        
        # Load model
        model = load_model(MODEL_PATH)
        
        # Load dataset
        X, y = load_test_dataset(DATASET_PATH, crop_w)
        
        if y is None:
            print("❌ Não é possível prosseguir sem a variável alvo")
            return
        
        # Calculate errors
        predictions, raw_errors, abs_errors = calculate_errors(model, X, y)
        
        # Create plots
        create_error_distribution_plot(abs_errors, OUTPUT_PATH)
        
        # Save statistics
        save_error_statistics(abs_errors, raw_errors, OUTPUT_PATH)
        
        # Save error data
        save_error_csv(abs_errors, raw_errors, OUTPUT_PATH)
        
        print(f"\n{'='*70}")
        print("✓ Análise concluída com sucesso!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
