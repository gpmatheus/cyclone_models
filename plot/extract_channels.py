"""
Script para extrair e salvar canais de imagens do arquivo HDF5
Lê uma imagem do arquivo TCIR-ATLN_EPAC_WPAC.h5 e salva cada canal como PNG separado.
"""

import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def load_image_from_h5(filepath, image_index=0):
    """
    Carrega uma imagem do arquivo HDF5.
    
    Args:
        filepath: Caminho para o arquivo HDF5
        image_index: Índice da imagem a ser carregada
    
    Returns:
        image: Array numpy com a imagem (shape: height, width, channels)
    """
    # TODO: Implementar a leitura da imagem do arquivo H5
    with h5py.File("data/TCIR-ALL_2017.h5", mode="r") as file:
        return file["matrix"][200, :, :, :]


def save_channels_as_png(image, image_index, output_dir):
    """
    Salva cada canal da imagem como um arquivo PNG separado.
    
    Args:
        image: Array numpy com a imagem (shape: height, width, channels)
        image_index: Índice da imagem (para identificação no nome do arquivo)
        output_dir: Diretório onde salvar os PNGs
    """
    if image is None:
        print("❌ Erro: Imagem não foi carregada")
        return
    
    # Verificar número de canais
    if len(image.shape) != 3:
        print(f"❌ Erro: Imagem deve ter 3 dimensões (H, W, C), mas tem shape {image.shape}")
        return
    
    height, width, num_channels = image.shape
    print(f"\nProcessando imagem {image_index}:")
    print(f"  Dimensões: {height} x {width}")
    print(f"  Canais: {num_channels}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Processar cada canal
    for channel_idx in range(num_channels):
        channel_data = image[:, :, channel_idx]
        
        # Normalizar para intervalo [0, 255]
        if channel_data.min() != channel_data.max():
            channel_normalized = ((channel_data - channel_data.min()) / 
                                 (channel_data.max() - channel_data.min()) * 255).astype(np.uint8)
        else:
            channel_normalized = np.zeros_like(channel_data, dtype=np.uint8)
        
        # Criar imagem PIL
        img = Image.fromarray(channel_normalized, mode='L')
        
        # Salvar com nome indicando canal
        output_filename = f"imagem_{image_index:04d}_canal_{channel_idx}.png"
        output_path = output_dir / output_filename
        img.save(output_path)
        
        print(f"  ✓ Canal {channel_idx} salvo: {output_filename}")
        print(f"    Min: {channel_data.min():.2f}, Max: {channel_data.max():.2f}, Mean: {channel_data.mean():.2f}")


def main():
    """Execução principal."""
    root_dir = Path(__file__).parent.parent
    
    # Caminhos
    h5_file = root_dir / "data" / "TCIR-ALL_2017.h5"
    output_dir = root_dir / "result" / "plot" / "channel_extraction"
    
    # Verificar se arquivo existe
    if not h5_file.exists():
        print(f"❌ Arquivo não encontrado: {h5_file}")
        return
    
    print("="*70)
    print("  EXTRAÇÃO DE CANAIS DE IMAGEM")
    print("="*70)
    
    # Carregar imagem
    print(f"\n[1/2] Carregando imagem de {h5_file}...")
    image_index = 0  # Primeiro índice (pode ser alterado)
    image = load_image_from_h5(str(h5_file), image_index)
    
    if image is None:
        print("❌ Falha ao carregar a imagem")
        return
    
    # Salvar canais como PNG
    print(f"\n[2/2] Salvando canais como PNG...")
    save_channels_as_png(image, image_index, output_dir)
    
    print(f"\n{'='*70}")
    print(f"✓ Canais salvos em: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
