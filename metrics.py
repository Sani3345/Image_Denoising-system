from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

clean = np.array(Image.open("clean.png")).astype(np.float32) / 255.0
noisy = np.array(Image.open("noisy.png")).astype(np.float32) / 255.0
denoised = np.array(Image.open("denoised.png")).astype(np.float32) / 255.0

noisy_psnr = peak_signal_noise_ratio(clean, noisy, data_range=1.0)
denoised_psnr = peak_signal_noise_ratio(clean, denoised, data_range=1.0)

noisy_ssim = structural_similarity(
    clean,
    noisy,
    channel_axis=2,
    data_range=1.0
)

denoised_ssim = structural_similarity(
    clean,
    denoised,
    channel_axis=2,
    data_range=1.0
)

print("\n===== DENOISING RESULTS =====")

print(f"Noisy PSNR:     {noisy_psnr:.2f} dB")
print(f"Denoised PSNR:  {denoised_psnr:.2f} dB")

print(f"Noisy SSIM:     {noisy_ssim:.4f}")
print(f"Denoised SSIM:  {denoised_ssim:.4f}")