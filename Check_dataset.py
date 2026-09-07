from PIL import Image
import os

folder = "/Users/saniyakumari/Image_Denoising1/Dataset.py/archive/DIV2K_train_HR/DIV2K_train_HR"

files = sorted([
    f for f in os.listdir(folder)
    if f.lower().endswith(".png")
])

print("Total images:", len(files))

if len(files) == 0:
    print("No PNG images found!")
    exit()

image_path = os.path.join(folder, files[0])

image = Image.open(image_path)

print("First image:", files[0])
print("Image size:", image.size)
print("Image mode:", image.mode)