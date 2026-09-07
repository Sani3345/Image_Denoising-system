from PIL import Image, ImageDraw

clean = Image.open("clean.png").convert("RGB")
noisy = Image.open("noisy.png").convert("RGB")
denoised = Image.open("denoised.png").convert("RGB")

width, height = clean.size
label_height = 30

comparison = Image.new(
    "RGB",
    (width * 3, height + label_height),
    "white"
)

comparison.paste(clean, (0, label_height))
comparison.paste(noisy, (width, label_height))
comparison.paste(denoised, (width * 2, label_height))

draw = ImageDraw.Draw(comparison)

draw.text((10, 8), "Clean", fill="black")
draw.text((width + 10, 8), "Noisy", fill="black")
draw.text((width * 2 + 10, 8), "Denoised", fill="black")

comparison.save("comparison.png")

print("Visual comparison saved as comparison.png")