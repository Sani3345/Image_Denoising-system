import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from div2k_dataset import DIV2KDataset
from model import DenoisingCNN


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


image_dir = "/Users/saniyakumari/Image_Denoising1/Dataset.py/archive/DIV2K_train_HR/DIV2K_train_HR"


dataset = DIV2KDataset(
    image_dir=image_dir,
    patch_size=128,
    patches_per_image=5,
    noise_std=25
)


train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size


train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)


train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)


print("Training patches:", len(train_dataset))
print("Validation patches:", len(val_dataset))


model = DenoisingCNN().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


epochs = 30

best_val_loss = float("inf")


for epoch in range(epochs):

    model.train()

    train_loss = 0.0

    for noisy, clean in train_loader:

        noisy = noisy.to(device)
        clean = clean.to(device)

        output = model(noisy)

        loss = criterion(output, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(train_loader)


    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for noisy, clean in val_loader:

            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)

            loss = criterion(output, clean)

            val_loss += loss.item()


    val_loss /= len(val_loader)

    scheduler.step(val_loss)


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Training Loss: {train_loss:.6f} "
        f"Validation Loss: {val_loss:.6f}"
    )


    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_denoising_model.pth"
        )

        print("Best model saved!")


print("Training completed!")