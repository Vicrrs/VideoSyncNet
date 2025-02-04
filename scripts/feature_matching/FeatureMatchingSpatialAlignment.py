import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
SIFT, SURF, ORB, BRISK → para detectar pontos de interesse.
RANSAC + Homografia → para alinhar os quadros.
"""

# Carregar as imagens em escala de cinza
img1 = cv2.imread('/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj002.jpg', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Erro ao carregar as imagens.")
    exit()

# Detectar pontos de interesse com ORB
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Criar o matcher de características (BFMatcher com cross-check)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Ordenar os matches com base na distância
matches = sorted(matches, key=lambda x: x.distance)

# Extrair os pontos correspondentes para o cálculo da homografia
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Estimar a homografia usando RANSAC
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Alinhar a segunda imagem à primeira
aligned_img2 = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

# Calcular a diferença absoluta entre a imagem 1 e a imagem 2 alinhada
diff = cv2.absdiff(img1, aligned_img2)

# Plotar os resultados com Matplotlib
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Imagem 1 (Original)
axes[0, 0].imshow(img1, cmap='gray')
axes[0, 0].set_title("Imagem 1 (Original)")
axes[0, 0].axis('off')

# Imagem 2 (Original)
axes[0, 1].imshow(img2, cmap='gray')
axes[0, 1].set_title("Imagem 2 (Original)")
axes[0, 1].axis('off')

# Imagem 2 Alinhada
axes[1, 0].imshow(aligned_img2, cmap='gray')
axes[1, 0].set_title("Imagem 2 Alinhada")
axes[1, 0].axis('off')

# Diferença entre Imagem 1 e Imagem 2 Alinhada
axes[1, 1].imshow(diff, cmap='gray')
axes[1, 1].set_title("Diferença Absoluta")
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
