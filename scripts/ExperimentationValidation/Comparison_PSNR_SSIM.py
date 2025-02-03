from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import cv2

# Carrega imagens (original e processada)
img_original = cv2.imread('original_frame.png', cv2.IMREAD_GRAYSCALE)
img_processada = cv2.imread('alinhado_frame.png', cv2.IMREAD_GRAYSCALE)

# Calcula PSNR
valor_psnr = psnr(img_original, img_processada)
print(f"PSNR: {valor_psnr:.2f} dB")

# Calcula SSIM
valor_ssim, _ = ssim(img_original, img_processada, full=True)
print(f"SSIM: {valor_ssim:.4f}")
