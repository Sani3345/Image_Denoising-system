# Image_Denoising-system
# Image Denoising using Deep Learning

This project is about removing Gaussian noise from images using a deep learning model built with PyTorch.

I trained a CNN-based image denoising model on the DIV2K dataset. The model takes a noisy image as input and tries to recover a cleaner version of the original image.

## What I used

* Python
* PyTorch
* Torchvision
* NumPy
* Pillow
* scikit-image
* Matplotlib

The model was trained using Apple Silicon MPS on my MacBook.

## Dataset

I used the DIV2K high-resolution training dataset, which contains 800 images.

For training, I used:

* Patch size: 128 × 128
* Gaussian noise: 25/255
* 5 random patches from each image
* Random horizontal and vertical flips
* Batch size: 8
* Training epochs: 30

The dataset itself is not included in this repository because of its large size.

## Model

The denoising network is a CNN inspired by the DnCNN architecture.

Instead of directly predicting the clean image, the network learns to estimate the noise present in the image. The predicted noise is then subtracted from the noisy input to obtain the denoised image.

The model uses convolutional layers, ReLU activations and batch normalization.

## Training

I used:

* Loss function: MSE Loss
* Optimizer: Adam
* Learning rate: 0.0005
* Learning rate scheduler: ReduceLROnPlateau
* Device: Apple MPS

The validation loss decreased during training, and the best model was saved based on the lowest validation loss.

## Results

I evaluated the model using PSNR and SSIM.

| Metric | Noisy Image | Denoised Image |
| ------ | ----------: | -------------: |
| PSNR   |    20.66 dB |       31.27 dB |
| SSIM   |      0.3475 |         0.8583 |

The average improvement was:

* PSNR: **+10.61 dB**
* SSIM: **+0.5108**

These results show that the model was able to remove a significant amount of the added Gaussian noise while preserving image details.

## Project Structure

```text
Image_Denoising-system/
│
├── model.py
├── div2k_dataset.py
├── train_model.py
├── evaluate_model.py
├── validate_model.py
├── test_model.py
├── test_dataset.py
├── process_full_dataset.py
├── metrics.py
├── visual_compare.py
├── denoise_test.py
├── Check_dataset.py
├── results_graph.py
│
├── psnr_comparison.png
├── ssim_comparison.png
├── comparison.png
├── clean.png
├── noisy.png
├── denoised.png
│
├── .gitignore
└── README.md
```

The dataset, generated image folders and model checkpoint files are excluded from GitHub because they are large.

## How to Run

First install the required packages:

```bash
pip install torch torchvision pillow numpy matplotlib scikit-image
```

Make sure the DIV2K dataset is available locally and update the dataset path in the code if needed.

Then run the training script:

```bash
python train_model.py
```

After training, the model can be evaluated using:

```bash
python evaluate_model.py
```

Other scripts in the repository can be used for testing the dataset, validating the model and comparing noisy and denoised images.

## Example

The basic workflow of the project is:

```text
Clean Image
     ↓
Add Gaussian Noise
     ↓
Noisy Image
     ↓
CNN Denoising Model
     ↓
Denoised Image
```

## Future Improvements

Some things I would like to try in the future are:

* Training for more epochs
* Testing different noise levels
* Trying different CNN architectures
* Using larger or more varied datasets
* Comparing the model with other denoising methods
* Improving inference speed

## Conclusion

This project helped me understand how deep learning can be used for image restoration. I learned about creating noisy training data, preparing image patches, training a CNN, evaluating image quality using PSNR and SSIM, and working with PyTorch on Apple Silicon.

## Author

**Saniya Kumari**

