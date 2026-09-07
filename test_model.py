import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from model import DenoisingCNN


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

image_dir = "/Users/saniyakumari/Image_Denoising1/Dataset.py/archive/DIV2K_train_HR/DIV2K_train_HR"

output_dir = "denoising_results"
os.makedirs(output_dir, exist_ok=True)

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


print("Total images:", len(image_files))
print()


def psnr(image1, image2):
    mse = torch.mean((image1 - image2) ** 2)

    if mse == 0:
        return float("inf")

    return 10 * torch.log10(1.0 / mse).item()


total_noisy_psnr = 0.0
total_denoised_psnr = 0.0


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


    noisy_score = psnr(
        noisy,
        clean
    )

    denoised_score = psnr(
        denoised,
        clean
    )


    total_noisy_psnr += noisy_score
    total_denoised_psnr += denoised_score


    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]


    save_image(
        clean,
        os.path.join(
            output_dir,
            f"{filename}_original.png"
        )
    )

    save_image(
        noisy,
        os.path.join(
            output_dir,
            f"{filename}_noisy.png"
        )
    )

    save_image(
        denoised,
        os.path.join(
            output_dir,
            f"{filename}_denoised.png"
        )
    )


    print(
        f"[{i + 1}/{len(image_files)}] "
        f"{filename} "
        f"Noisy: {noisy_score:.2f} dB "
        f"Denoised: {denoised_score:.2f} dB"
    )


average_noisy_psnr = (
    total_noisy_psnr / len(image_files)
)

average_denoised_psnr = (
    total_denoised_psnr / len(image_files)
)

improvement = (
    average_denoised_psnr - average_noisy_psnr
)


print()
print("===================================")
print(
    f"Average Noisy PSNR: "
    f"{average_noisy_psnr:.2f} dB"
)
print(
    f"Average Denoised PSNR: "
    f"{average_denoised_psnr:.2f} dB"
)
print(
    f"PSNR Improvement: "
    f"{improvement:.2f} dB"
)
print("===================================")
print()
print("All results saved in:", output_dir)