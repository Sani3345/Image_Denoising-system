import matplotlib.pyplot as plt


labels = ["Noisy", "Denoised"]

psnr = [20.66, 31.27]
ssim = [0.3475, 0.8583]


plt.figure()
plt.bar(labels, psnr)
plt.ylabel("PSNR (dB)")
plt.title("PSNR Comparison")
plt.savefig("psnr_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure()
plt.bar(labels, ssim)
plt.ylabel("SSIM")
plt.title("SSIM Comparison")
plt.savefig("ssim_comparison.png", dpi=300, bbox_inches="tight")
plt.show()