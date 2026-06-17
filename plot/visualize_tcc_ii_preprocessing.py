"""
Visualization of TCC-II preprocessing technique
Shows the two-step differencing calculation for cyclone motion detection
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def load_sample_images(dataset_path: str, n_samples: int = 3) -> np.ndarray:
    """
    Load n_samples consecutive images from HDF5 dataset.
    
    Args:
        dataset_path: Path to HDF5 file
        n_samples: Number of consecutive images to load
        
    Returns:
        Array of shape (n_samples, H, W) or (n_samples, H, W, C)
    """
    with h5py.File(dataset_path, 'r') as f:
        images = f['matrix']

        imgs_len = images.shape[0]

        import random

        # idx = random.randint(4, imgs_len - 4)
        idx = 26524

        print(f'idx: {idx}')
        
        return images[idx:idx + n_samples, :, :, :]


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] for display only."""
    img_min = np.nanmin(img)
    img_max = np.nanmax(img)
    
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img.astype(np.float32)


def create_visualization(images: np.ndarray, output_dir: str = "result/plot"):
    """
    Create 3 separate visualizations of TCC-II preprocessing pipeline.
    Each image saved individually in separate subfolders.
    
    Args:
        images: Array of consecutive images (4, H, W) or (4, H, W, C)
        output_dir: Directory to save the visualizations
    """
    # Ensure output directories exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dir1 = Path(output_dir) / "01_original_images"
    dir2 = Path(output_dir) / "02_differences"
    dir3 = Path(output_dir) / "03_result"
    dir1.mkdir(exist_ok=True)
    dir2.mkdir(exist_ok=True)
    dir3.mkdir(exist_ok=True)
    
    # Extract 3 consecutive images
    if len(images) < 3:
        raise ValueError(f"Need at least 4 images, got {len(images)}")
    
    # img_n_minus_2 = images[0]
    # img_n_minus_1 = images[1]
    # img_n = images[2]
    # img_n_plus_1 = images[3]

    img_n = images[2]
    img_n_minus_1 = images[1]
    img_n_minus_2 = images[0]
    
    # Handle multi-channel: take first channel only
    if len(img_n_minus_2.shape) == 3:
        img_n_minus_2 = img_n_minus_2[:, :, 0]
    if len(img_n_minus_1.shape) == 3:
        img_n_minus_1 = img_n_minus_1[:, :, 0]
    if len(img_n.shape) == 3:
        img_n = img_n[:, :, 0]
    
    # Calculate intermediate differences
    diff_n_minus_1 = img_n_minus_1 - img_n_minus_2  # (I^n-1 - I^n-2)
    diff_n = img_n - img_n_minus_1                   # (I^n - I^n-1)
    
    # Calculate final second derivative
    final_diff = np.abs(diff_n - diff_n_minus_1)    # |(I^n - I^n-1) - (I^n-1 - I^n-2)|
    
    # ========== FOLDER 1: Original Images ==========
    imgs_to_plot = [
        (img_n_minus_2, "I_n-2"),
        (img_n_minus_1, "I_n-1"),
        (img_n, "I_n"),
    ]
    
    for img, label in imgs_to_plot:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.imshow(normalize_for_display(img), cmap='viridis')
        # ax.set_title(label, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        path = str(dir1 / f"{label}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {path}")
        plt.close()
    
    # ========== FOLDER 2: Differences ==========
    diffs_to_plot = [
        (diff_n_minus_1, "Diff_n-1_minus_n-2"),
        (diff_n, "Diff_n_minus_n-1"),
    ]
    
    for img, label in diffs_to_plot:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ax.imshow(normalize_for_display(img), cmap='RdBu_r')
        # ax.set_title(label.replace('_', ' '), fontsize=12, fontweight='bold')
        ax.axis('off')
        
        path = str(dir2 / f"{label}.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {path}")
        plt.close()
    
    # ========== FOLDER 3: Final Result ==========
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.imshow(normalize_for_display(final_diff), cmap='hot')
    # ax.set_title("Final Result", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    path = str(dir3 / "result.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved {path}")
    plt.close()
    
    # Print statistics
    # print("\n" + "="*70)
    # print("  TCC-II PREPROCESSING STATISTICS")
    # print("="*70)
    # print(f"\nInput Images Shape: {img_n.shape}")
    # print(f"Data Type: {img_n.dtype}")
    # print(f"\nOriginal Image Statistics:")
    # print(f"  I^(n-2): min={np.nanmin(img_n_minus_2):.4f}, max={np.nanmax(img_n_minus_2):.4f}, mean={np.nanmean(img_n_minus_2):.4f}")
    # print(f"  I^(n-1): min={np.nanmin(img_n_minus_1):.4f}, max={np.nanmax(img_n_minus_1):.4f}, mean={np.nanmean(img_n_minus_1):.4f}")
    # print(f"  I^(n):   min={np.nanmin(img_n):.4f}, max={np.nanmax(img_n):.4f}, mean={np.nanmean(img_n):.4f}")
    # print(f"  I^(n+1): min={np.nanmin(img_n_plus_1):.4f}, max={np.nanmax(img_n_plus_1):.4f}, mean={np.nanmean(img_n_plus_1):.4f}")
    
    # print(f"\nFirst Level Differences:")
    # print(f"  Diff1a (I^(n-1) - I^(n-2)): min={np.nanmin(diff_n_minus_1):.4f}, max={np.nanmax(diff_n_minus_1):.4f}")
    # print(f"  Diff1b (I^(n) - I^(n-1)):   min={np.nanmin(diff_n):.4f}, max={np.nanmax(diff_n):.4f}")
    
    # print(f"\nFinal Second-Order Difference:")
    # print(f"  Δ²I: min={np.nanmin(final_diff):.4f}, max={np.nanmax(final_diff):.4f}, mean={np.nanmean(final_diff):.4f}")
    # print(f"      (Measures motion acceleration/deceleration)")
    # print("="*70 + "\n")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("  TCC-II Preprocessing Visualization")
    print("="*70)
    
    # Paths
    data_file = "data/TCIR-ATLN_EPAC_WPAC.h5"
    output_dir = "result/plot"
    
    # Load images
    print(f"\nLoading sample images from {data_file}...")
    try:
        images = load_sample_images(data_file, n_samples=4)
        print(f"Loaded images shape: {images.shape}")
        print(f"Data type: {images.dtype}")
    except Exception as e:
        print(f"❌ Error loading images: {e}")
        return
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    try:
        create_visualization(images, output_dir)
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("✓ Done!")


if __name__ == "__main__":
    main()
