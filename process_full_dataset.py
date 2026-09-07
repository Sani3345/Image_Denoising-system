import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from model import DenoisingCNN

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

clean_dir = "/Users/saniyakumari/Image_Denoising1/Dataset.py/archive/DIV2K_train_HR/DIV2K_train_HR"

noisy_dir = "noisy_dataset"
denoised_dir = "denoised_dataset"

os.makedirs(noisy_dir, exist_ok=True)
os.makedirs(denoised_dir, exist_ok=True)

model = DenoisingCNN().to(device)

model.load_state_dict(
    torch.load(
        "best_denoising_model.pth",
        map_location=device
    )
)

model.eval()

print("Best model loaded!")

image_files = sorted([
    f for f in os.listdir(clean_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

print("Total images:", len(image_files))

noise_std = 50

print("Noise standard deviation:", noise_std)


for i, filename in enumerate(image_files):

    image_path = os.path.join(clean_dir, filename)


    image = Image.open(image_path).convert("RGB")

    clean = TF.to_tensor(image)


    noise = torch.randn_like(clean) * (noise_std / 255.0)

    noisy = torch.clamp(
        clean + noise,
        0.0,
        1.0
    )


    noisy_input = noisy.unsqueeze(0).to(device)


    with torch.no_grad():

        denoised = model(noisy_input)


    denoised = torch.clamp(
        denoised.squeeze(0).cpu(),
        0.0,
        1.0
    )


    noisy_path = os.path.join(
        noisy_dir,
        filename
    )

    TF.to_pil_image(noisy).save(noisy_path)


    denoised_path = os.path.join(
        denoised_dir,
        filename
    )

    TF.to_pil_image(denoised).save(denoised_path)


    print(
        f"Processed {i + 1}/{len(image_files)}: {filename}"
    )


print("\n========================================")
print("Full dataset processing completed!")
print("========================================")

print("Noise level:", noise_std)

print("Noisy images saved in:")
print(noisy_dir)

print("Denoised images saved in:")
print(denoised_dir)