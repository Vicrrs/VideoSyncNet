import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DeepVisionCleaner import ImagePreprocessor
from feature_extractor import FeatureExtractor
from spatial_temporal_alignment import SpatialTemporalAligner

# Import para SSIM
from skimage.metrics import structural_similarity as ssim

FOLDER = "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/garrafa"
OUTPUT_DIR = "output_garrafa_faca02"
os.makedirs(OUTPUT_DIR, exist_ok=True)

valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
image_files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith(valid_exts)])

preprocessor = ImagePreprocessor(
    apply_gaussian=True,
    apply_bilateral=True,
    apply_median=True,
    apply_clahe=True,
    apply_perspective_correction=False
)

extractor = FeatureExtractor(
    use_cnn=False,
    use_hog=False,
    use_optical_flow=True,
    use_siamese=True
)

aligner = SpatialTemporalAligner(
    use_epipolar=False,
    use_warping=True,
    use_siamese=True
)

distancias_raw = []
distancias_alinhadas = []
ssim_raw = []
ssim_aligned = []
frame_indices = []

for i in range(len(image_files) - 1):
    img1_path = os.path.join(FOLDER, image_files[i])
    img2_path = os.path.join(FOLDER, image_files[i + 1])

    img1_raw = cv2.imread(img1_path)
    img2_raw = cv2.imread(img2_path)

    if img1_raw is None or img2_raw is None:
        print(f"Falha ao carregar: {img1_path} ou {img2_path}. Pulando...")
        continue

    # Pré-processa
    img1_proc = preprocessor.preprocess_image(img1_path)
    img2_proc = preprocessor.preprocess_image(img2_path)

    # Distância siamesa (sem warp)
    dist_raw = extractor.compute_siamese_distance(img1_proc, img2_proc)

    # Alinha
    aligned1, aligned2 = aligner.align_frames(img1_proc, img2_proc)

    # Distância siamesa (com warp)
    dist_aligned = aligner.compute_siamese_similarity(img1_proc, aligned2)

    # --- Calcula SSIM ---
    # Converte para escala de cinza
    gray1_proc = cv2.cvtColor(img1_proc, cv2.COLOR_BGR2GRAY)
    gray2_proc = cv2.cvtColor(img2_proc, cv2.COLOR_BGR2GRAY)
    gray_aligned2 = cv2.cvtColor(aligned2, cv2.COLOR_BGR2GRAY)

    score_raw, _ = ssim(gray1_proc, gray2_proc, full=True)
    score_aligned, _ = ssim(gray1_proc, gray_aligned2, full=True)

    # Armazena nos vetores
    distancias_raw.append(dist_raw)
    distancias_alinhadas.append(dist_aligned)
    ssim_raw.append(score_raw)
    ssim_aligned.append(score_aligned)
    frame_indices.append(i)

    print(f"Par {i}-{i+1}:")
    print(f"  Distância Siamese (sem warp) = {dist_raw:.2f}, (com warp) = {dist_aligned:.2f}")
    print(f"  SSIM (sem warp) = {score_raw:.3f}, (com warp) = {score_aligned:.3f}")

# --- Gera gráficos ---
plt.figure(figsize=(9, 4))
plt.subplot(1,2,1)
plt.plot(frame_indices, distancias_raw, marker='o', label='Siamese - Sem Warp')
plt.plot(frame_indices, distancias_alinhadas, marker='o', label='Siamese - Warp')
plt.xlabel("Par de Imagens")
plt.ylabel("Dist. Siamesa")
plt.title("Distância Siamesa: Sem Warp vs Warp")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(frame_indices, ssim_raw, marker='o', label='SSIM - Sem Warp')
plt.plot(frame_indices, ssim_aligned, marker='o', label='SSIM - Warp')
plt.xlabel("Par de Imagens")
plt.ylabel("SSIM")
plt.title("SSIM: Sem Warp vs Warp")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("comparacao_siamese_ssim.png")
print("Gráfico salvo em comparacao_siamese_ssim.png")
