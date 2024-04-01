import matplotlib.pyplot as plt
import numpy as np

from skimage import data, img_as_float
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


img = img_as_float(data.camera())
rows, cols = img.shape

noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
rng = np.random.default_rng()
noise[rng.random(size=noise.shape) > 0.5] *= -1

img_noise = img + noise
img_const = img + abs(noise)

img_3 = np.stack((img, img, img), axis=-1)
img_noise_3 = np.stack((img_noise, img_noise, img_noise), axis=-1)
img_const_3 = np.stack((img_const, img_const, img_const), axis=-1)

print("image:", img.shape, img.min(), img.max())
print("noise:", noise.shape, noise.min(), noise.max())
print("img_noise:", img_noise.shape, img_noise.min(), img_noise.max())
print("img_const:", img_const.shape, img_const.min(), img_const.max())
print("img_3:", img_3.shape, img_3.min(), img_3.max())
print("img_noise_3:", img_noise_3.shape, img_noise_3.min(), img_noise_3.max())
print("img_const_3:", img_const_3.shape, img_const_3.min(), img_const_3.max())

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4), sharex=True, sharey=True)
ax = axes.ravel()

mse_none = mean_squared_error(img, img)
ssim_none = ssim(img, img, data_range=img.max() - img.min())

mse_noise = mean_squared_error(img, img_noise)
ssim_noise = ssim(img, img_noise, data_range=img_noise.max() - img_noise.min())
print("SSIM (noise):", ssim_noise)

mse_const = mean_squared_error(img, img_const)
ssim_const = ssim(img, img_const, data_range=img_const.max() - img_const.min())


ssim_noise_3 = ssim(
    img_3, img_noise_3, multi_channel=True, channel_axis=-1, data_range=img_noise_3.max() - img_noise_3.min()
)
print("SSIM (noise):", ssim_noise_3)

ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(f"MSE: {mse_none:.2f}, SSIM: {ssim_none:.2f}")
ax[0].set_title("Original image")

ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(f"MSE: {mse_noise:.2f}, SSIM: {ssim_noise:.2f}")
ax[1].set_title("Image with noise")

ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[2].set_xlabel(f"MSE: {mse_const:.2f}, SSIM: {ssim_const:.2f}")
ax[2].set_title("Image plus constant")

plt.tight_layout()
plt.show()
