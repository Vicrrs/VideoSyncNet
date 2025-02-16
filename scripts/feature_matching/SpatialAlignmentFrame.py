import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar as imagens em escala de cinza
img1 = cv2.imread('/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj002.jpg', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Erro ao carregar as imagens.")
    exit()

# Inicializar a matriz de transformação (identidade)
warp_matrix = np.eye(2, 3, dtype=np.float32)

# Definir os critérios de término para o algoritmo ECC
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)

# Executar o algoritmo ECC para estimar a transformação
(cc, warp_matrix) = cv2.findTransformECC(img1, img2, warp_matrix, cv2.MOTION_AFFINE, criteria)

# Aplicar a transformação na imagem 2 para alinhá-la à imagem 1
aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]))

# Exibir as imagens utilizando o Matplotlib
plt.figure(figsize=(15, 5))

# Imagem de referência
plt.subplot(1, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title("Imagem 1 (Referência)")
plt.axis('off')

# Imagem original
plt.subplot(1, 3, 2)
plt.imshow(img2, cmap='gray')
plt.title("Imagem 2 (Original)")
plt.axis('off')

# Imagem alinhada
plt.subplot(1, 3, 3)
plt.imshow(aligned, cmap='gray')
plt.title("Imagem 2 (Alinhada)")
plt.axis('off')

plt.tight_layout()
plt.show()
