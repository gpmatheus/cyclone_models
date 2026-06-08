"""
Model Error Analysis Script
Loads a trained model and calculates prediction errors on test dataset.
Generates frequency distribution plots of absolute errors.

Environment variables:
  MODEL_PATH: Path to the trained model file (.keras or .h5)
  DATASET_PATH: Path to the test dataset (HDF5 file)
  OUTPUT_PATH: Directory to save results
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
    print("⚠️  TensorFlow/Keras not available. Install with: pip install tensorflow")


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
    """Calculate prediction errors for each sample."""
    print(f"\nMaking predictions on {len(X)} samples...")
    
    try:
        # Make predictions
        predictions = model.predict(X, verbose=0)
        
        # Handle multi-dimensional predictions (take first output if needed)
        if len(predictions.shape) > 1:
            predictions = predictions[:, 0]
        
        print(f"✓ Predictions completed")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Predictions range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
        
        if y is not None:
            # Calculate raw errors
            raw_errors = y - predictions
            
            # Calculate absolute errors
            abs_errors = np.abs(raw_errors)
            
            print(f"\n✓ Errors calculated")
            print(f"  Raw errors - mean: {np.mean(raw_errors):.4f}, std: {np.std(raw_errors):.4f}")
            print(f"  Absolute errors - mean: {np.mean(abs_errors):.4f}, std: {np.std(abs_errors):.4f}")
            print(f"  Absolute errors - min: {np.min(abs_errors):.4f}, max: {np.max(abs_errors):.4f}")
            
            return predictions, raw_errors, abs_errors
        else:
            return predictions, None, None
            
    except Exception as e:
        raise RuntimeError(f"Failed to make predictions: {e}")


def create_error_distribution_plot(abs_errors, output_path):
    """Create frequency distribution plot of absolute errors."""
    print(f"\nCreating error distribution plot...")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    n, bins, patches = ax1.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Absolute Prediction Errors', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add mean line
    mean_val = np.mean(abs_errors)
    ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, label=f'Mean: {mean_val:.4f}')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Add statistics text
    stats_text = (
        f"Mean: {np.mean(abs_errors):.4f}\n"
        f"Median: {np.median(abs_errors):.4f}\n"
        f"Std Dev: {np.std(abs_errors):.4f}\n"
        f"Min: {np.min(abs_errors):.4f}\n"
        f"Max: {np.max(abs_errors):.4f}\n"
        f"Count: {len(abs_errors)}"
    )
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontfamily='monospace', fontsize=9)
    
    # Cumulative distribution
    ax2 = axes[1]
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax2.plot(sorted_errors, cumulative, linewidth=2, color='steelblue')
    ax2.set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax2.set_title('Cumulative Distribution of Absolute Errors', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_path) / "error_distribution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {output_file}")
    plt.close()


def save_error_statistics(abs_errors, raw_errors, output_path):
    """Save error statistics as JSON."""
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
    print(f"✓ Statistics saved to {output_file}")


def save_error_csv(abs_errors, raw_errors, output_path):
    """Save errors to CSV file for further analysis."""
    import csv
    
    output_file = Path(output_path) / "errors.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_ID', 'Raw_Error', 'Absolute_Error'])
        for idx, (raw_err, abs_err) in enumerate(zip(raw_errors, abs_errors)):
            writer.writerow([idx, raw_err, abs_err])
    
    print(f"✓ Errors saved to {output_file}")


def main(crop_w=64):
    """Main execution."""
    try:
        
        # Load model
        model = load_model(MODEL_PATH)
        
        # Load dataset
        X, y = load_test_dataset(DATASET_PATH, crop_w)
        
        if y is None:
            print("❌ Cannot proceed without target variable")
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
        print("✓ Analysis completed successfully!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
