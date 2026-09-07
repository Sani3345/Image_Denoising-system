import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from model import DenoisingCNN


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


image_dir = "/Users/saniyakumari/Image_Denoising1/Dataset.py/archive/DIV2K_train_HR/DIV2K_train_HR"

model = DenoisingCNN().to(device)

model.load_state_dict(
    torch.load(
        "best_denoising_model.pth",
        map_location=device
    )
)

model.eval()


image_files = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

val_files = image_files[-80:]

total_mse_noisy = 0.0
total_mse_denoised = 0.0


with torch.no_grad():

    for image_path in val_files:

        image = Image.open(image_path).convert("RGB")

        clean = TF.to_tensor(image)

        noise = torch.randn_like(clean) * (25 / 255.0)

        noisy = torch.clamp(
            clean + noise,
            0.0,
            1.0
        )

        _, height, width = clean.shape

        crop_height = (height // 128) * 128
        crop_width = (width // 128) * 128

        clean = clean[:, :crop_height, :crop_width]
        noisy = noisy[:, :crop_height, :crop_width]

        denoised = torch.zeros_like(clean)

        for top in range(0, crop_height, 128):
            for left in range(0, crop_width, 128):

                noisy_patch = noisy[
                    :,
                    top:top + 128,
                    left:left + 128
                ]

                patch = noisy_patch.unsqueeze(0).to(device)

                output = model(patch)

                denoised[
                    :,
                    top:top + 128,
                    left:left + 128
                ] = output.squeeze(0).cpu()

        mse_noisy = F.mse_loss(noisy, clean).item()
        mse_denoised = F.mse_loss(denoised, clean).item()

        total_mse_noisy += mse_noisy
        total_mse_denoised += mse_denoised


avg_mse_noisy = total_mse_noisy / len(val_files)
avg_mse_denoised = total_mse_denoised / len(val_files)

psnr_noisy = 10 * torch.log10(
    torch.tensor(1.0 / avg_mse_noisy)
).item()

psnr_denoised = 10 * torch.log10(
    torch.tensor(1.0 / avg_mse_denoised)
).item()


print()
print("==============================")
print("Validation Results")
print("==============================")
print(f"Noisy PSNR: {psnr_noisy:.2f} dB")
print(f"Denoised PSNR: {psnr_denoised:.2f} dB")
print(f"PSNR Improvement: {psnr_denoised - psnr_noisy:.2f} dB")
print("==============================")