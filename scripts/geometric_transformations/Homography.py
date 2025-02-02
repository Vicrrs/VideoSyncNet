import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
img = cv2.imread('/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg')
rows, cols, _ = img.shape

# Definindo 4 pontos na imagem original
pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
# Definindo 4 pontos na imagem transformada (nova perspectiva)
pts2 = np.float32([[30, 70], [220, 50], [50, 250], [200, 230]])

# Calcular a matriz de homografia
H, _ = cv2.findHomography(pts1, pts2)

# Aplicar a transformação
img_homography = cv2.warpPerspective(img, H, (cols, rows))

# Exibir as imagens
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_homography, cv2.COLOR_BGR2RGB))
plt.title("Imagem com Homografia")
plt.axis("off")
plt.show()
