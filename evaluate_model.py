import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity

from model import DenoisingCNN


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

image_dir = "/Users/saniyakumari/Image_Denoising1/Dataset.py/archive/DIV2K_train_HR/DIV2K_train_HR"

model = DenoisingCNN().to(device)

model.load_state_dict(
    torch.load(
        "best_denoising_model.pth",
        map_location=device
    )
)

model.eval()


def psnr(image1, image2):
    mse = torch.mean((image1 - image2) ** 2)

    if mse == 0:
        return float("inf")

    return 10 * torch.log10(1.0 / mse).item()


def ssim(image1, image2):
    image1 = image1.permute(1, 2, 0).numpy()
    image2 = image2.permute(1, 2, 0).numpy()

    return structural_similarity(
        image1,
        image2,
        channel_axis=2,
        data_range=1.0
    )


image_files = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])


total_noisy_psnr = 0.0
total_denoised_psnr = 0.0
total_noisy_ssim = 0.0
total_denoised_ssim = 0.0


print("Total images:", len(image_files))
print()


for i, image_path in enumerate(image_files):

    image = Image.open(image_path).convert("RGB")

    clean = TF.to_tensor(image)

    height = clean.shape[1]
    width = clean.shape[2]

    new_height = height - height % 128
    new_width = width - width % 128

    clean = clean[:, :new_height, :new_width]

    noisy = torch.clamp(
        clean + torch.randn_like(clean) * (25 / 255.0),
        0.0,
        1.0
    )

    denoised = torch.zeros_like(noisy)

    height = noisy.shape[1]
    width = noisy.shape[2]


    for top in range(0, height, 128):

        for left in range(0, width, 128):

            patch = noisy[
                :,
                top:top + 128,
                left:left + 128
            ]

            patch = patch.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(patch)

            output = output.squeeze(0).cpu()

            denoised[
                :,
                top:top + 128,
                left:left + 128
            ] = output


    denoised = torch.clamp(
        denoised,
        0.0,
        1.0
    )


    noisy_psnr = psnr(noisy, clean)
    denoised_psnr = psnr(denoised, clean)

    noisy_ssim = ssim(noisy, clean)
    denoised_ssim = ssim(denoised, clean)


    total_noisy_psnr += noisy_psnr
    total_denoised_psnr += denoised_psnr

    total_noisy_ssim += noisy_ssim
    total_denoised_ssim += denoised_ssim


    print(
        f"[{i + 1}/{len(image_files)}] "
        f"{os.path.basename(image_path)} "
        f"PSNR: {denoised_psnr:.2f} dB "
        f"SSIM: {denoised_ssim:.4f}"
    )


count = len(image_files)

average_noisy_psnr = total_noisy_psnr / count
average_denoised_psnr = total_denoised_psnr / count

average_noisy_ssim = total_noisy_ssim / count
average_denoised_ssim = total_denoised_ssim / count


print()
print("======================================")
print(f"Average Noisy PSNR: {average_noisy_psnr:.2f} dB")
print(f"Average Denoised PSNR: {average_denoised_psnr:.2f} dB")
print(f"PSNR Improvement: {average_denoised_psnr - average_noisy_psnr:.2f} dB")
print()
print(f"Average Noisy SSIM: {average_noisy_ssim:.4f}")
print(f"Average Denoised SSIM: {average_denoised_ssim:.4f}")
print(f"SSIM Improvement: {average_denoised_ssim - average_noisy_ssim:.4f}")
print("======================================")