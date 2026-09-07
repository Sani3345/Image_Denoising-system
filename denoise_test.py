import torch
from PIL import Image
import torchvision.transforms.functional as TF
from model import DenoisingCNN

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

model = DenoisingCNN().to(device)

model.load_state_dict(
    torch.load(
    "best_denoising_model.pth",
    map_location=device

    )
)

model.eval()

print("Trained model loaded!")

image_path = "/Users/saniyakumari/Image_Denoising1/Dataset.py/archive/DIV2K_train_HR/DIV2K_train_HR/0001.png"

image = Image.open(image_path).convert("RGB")

clean = TF.to_tensor(image)

clean = clean[:, :128, :128]

noise_std = 25
noise = torch.randn_like(clean) * (noise_std / 255.0)

noisy = torch.clamp(clean + noise, 0.0, 1.0)

noisy_input = noisy.unsqueeze(0).to(device)

with torch.no_grad():
    denoised = model(noisy_input)

denoised = torch.clamp(denoised, 0.0, 1.0)

denoised = denoised.squeeze(0).cpu()

TF.to_pil_image(clean).save("clean.png")
TF.to_pil_image(noisy).save("noisy.png")
TF.to_pil_image(denoised).save("denoised.png")


print("Clean image saved as: clean.png")
print("Noisy image saved as: noisy.png")
print("Denoised image saved as: denoised.png")
print("Denoising test completed!")