
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt

def compute_optical_flow(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Optical flow usando Farneback (rápido e bom para deep learning)
    flow = cv2.calcOpticalFlowFarneback(
        img1, img2,
        None,
        pyr_scale=.5,
        levels=5,
        winsize=30,
        iterations=4,
        poly_n=50,
        poly_sigma=1.0,
        flags=0
    )

    # flow tem shape (H, W, 2): componente horizontal e vertical
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # magnitude e direção
    # magnitude = np.sqrt(u**2 + v**2)
    # direction = np.arctan(v, u)

    # return magnitude, direction
    return u, v


# open images

with h5py.File("data/TCIR-ATLN_EPAC_WPAC.h5", mode="r") as file:
    image1 = file['matrix'][100, :, :, 1]
    image2 = file['matrix'][101, :, :, 1]

print(image1.shape)
print(image2.shape)


# calculate optical flow
u, v = compute_optical_flow(image1, image2)


slc = slice(0, -1, 4)
image1 = image1[slc, slc]
image2 = image2[slc, slc]
u, v = u[slc, slc], v[slc, slc]

# vectors = np.concatenate((u, v), axis=-1)
x_origins = np.tile(np.arange(u.shape[0]), u.shape[1]).reshape(image1.shape)
y_origins = np.repeat(np.arange(v.shape[0]), v.shape[1]).reshape(image1.shape)

# u = u.flatten()
# v = v.flatten()

# direction = np.arctan2(v, u)

# fig, axes = plt.subplots(1, 3, figsize=(10, 5))

# axes[0].imshow(image1, interpolation="nearest")
# axes[1].imshow(image2, interpolation="nearest")

# axes[2].imshow(v, interpolation="nearest")

fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(image1, interpolation="nearest")

axes[1].imshow(image2, interpolation="nearest")

# desenha setas
axes[2].quiver(x_origins, y_origins, u, v, angles="xy", scale_units="xy", scale=1)

# magnitudes = np.sqrt(u ** 2 + v ** 2)
# axes[2].imshow(magnitudes, interpolation="nearest")


plt.show()

