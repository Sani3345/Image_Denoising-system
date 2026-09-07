from div2k_dataset import DIV2KDataset

image_dir = "/Users/saniyakumari/Image_Denoising1/Dataset.py/archive/DIV2K_train_HR/DIV2K_train_HR"

dataset = DIV2KDataset(
    image_dir=image_dir,
    patch_size=128,
    noise_std=25
)

noisy, clean = dataset[0]

print("Noisy shape:", noisy.shape)
print("Clean shape:", clean.shape)

print("Noisy minimum:", noisy.min().item())
print("Noisy maximum:", noisy.max().item())

print("Clean minimum:", clean.min().item())
print("Clean maximum:", clean.max().item())