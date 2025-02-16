import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sem interface gráfica
import matplotlib.pyplot as plt

from DeepVisionCleaner import ImagePreprocessor
from feature_extractor import FeatureExtractor
from spatial_temporal_alignment import SpatialTemporalAligner

# Caminho para a pasta com imagens da garrafa (e eventualmente faca)
FOLDER = "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/garrafa"
OUTPUT_DIR = "output_garrafa_faca"
os.makedirs(OUTPUT_DIR, exist_ok=True)

valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
image_files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith(valid_exts)])

# Instancia o pré-processador e os módulos
preprocessor = ImagePreprocessor(
    apply_gaussian=True,
    apply_bilateral=True,
    apply_median=True,
    apply_clahe=True,
    apply_perspective_correction=False
)

extractor = FeatureExtractor(
    use_cnn=False,        # Desligamos CNN para focar em Optical Flow e Siamese
    use_hog=False,
    use_optical_flow=True,
    use_siamese=True
)

aligner = SpatialTemporalAligner(
    use_epipolar=False,
    use_warping=True,     # Alinhamento com ORB+Homografia
    use_siamese=True
)

distancias_raw = []
distancias_alinhadas = []
frame_indices = []

for i in range(len(image_files) - 1):
    img1_path = os.path.join(FOLDER, image_files[i])
    img2_path = os.path.join(FOLDER, image_files[i + 1])

    # Carrega imagens originais
    img1_raw = cv2.imread(img1_path)
    img2_raw = cv2.imread(img2_path)

    if img1_raw is None or img2_raw is None:
        print(f"Falha ao carregar: {img1_path} ou {img2_path}. Pulando...")
        continue

    # Pré-processa as imagens para remover ruído, equalizar etc.
    img1_proc = preprocessor.preprocess_image(img1_path)
    img2_proc = preprocessor.preprocess_image(img2_path)

    # (1) Fluxo Óptico antes do warp
    flow_raw = extractor.compute_optical_flow(img1_proc, img2_proc)
    flow_raw_mag = np.sqrt(flow_raw[..., 0]**2 + flow_raw[..., 1]**2)

    # (2) Distância siamesa (sem warp)
    dist_raw = extractor.compute_siamese_distance(img1_proc, img2_proc)

    # (3) Alinhamento com ORB+Homografia
    aligned1, aligned2 = aligner.align_frames(img1_proc, img2_proc)

    # (4) Fluxo Óptico depois do warp (comparando img1_proc com aligned2)
    flow_aligned = extractor.compute_optical_flow(img1_proc, aligned2)
    flow_aligned_mag = np.sqrt(flow_aligned[..., 0]**2 + flow_aligned[..., 1]**2)

    # (5) Distância siamesa após warp
    dist_aligned = aligner.compute_siamese_similarity(img1_proc, aligned2)

    distancias_raw.append(dist_raw)
    distancias_alinhadas.append(dist_aligned)
    frame_indices.append(i)

    print(f"Par {i}-{i+1} | Distância sem warp={dist_raw:.2f} | Após warp={dist_aligned:.2f}")

    # --- Visualização ---
    # Cria uma imagem de debug: 2 linhas, 3 colunas
    #  1a linha: [img1_proc, img2_proc, flow_raw_mag]
    #  2a linha: [img1_proc, aligned2, flow_aligned_mag]

    # Aplica colorMap ao fluxo p/ visualização
    flow_raw_color = cv2.applyColorMap(
        cv2.convertScaleAbs(flow_raw_mag, alpha=0.5), cv2.COLORMAP_JET
    )
    flow_aligned_color = cv2.applyColorMap(
        cv2.convertScaleAbs(flow_aligned_mag, alpha=0.5), cv2.COLORMAP_JET
    )

    # Redimensiona se quiser, para uniformizar
    # (exemplo: se imagens são grandes)
    # ...
    
    # Empilha horizontalmente
    top_row = np.hstack([img1_proc, img2_proc, flow_raw_color])
    bottom_row = np.hstack([img1_proc, aligned2, flow_aligned_color])
    debug_image = np.vstack([top_row, bottom_row])

    # Desenha infos
    cv2.putText(debug_image,
                f"raw={dist_raw:.2f} warp={dist_aligned:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Salva debug
    out_file = os.path.join(OUTPUT_DIR, f"debug_{i:03d}.png")
    cv2.imwrite(out_file, debug_image)
    print(f"Salvo: {out_file}")

# Ao final, gera um gráfico das distâncias
plt.figure(figsize=(8,5))
plt.plot(frame_indices, distancias_raw, marker='o', label='Dist. Siamesa - Sem Warp')
plt.plot(frame_indices, distancias_alinhadas, marker='o', label='Dist. Siamesa - Warp')
plt.xlabel("Par de Imagens (índice)")
plt.ylabel("Distância Siamesa")
plt.title("Comparação: Sem Warp vs Alinhado (ORB+Homografia)")
plt.legend()
plt.grid(True)
plt.savefig("comparacao_siamesa.png")
print("Gráfico salvo em comparacao_siamesa.png")
