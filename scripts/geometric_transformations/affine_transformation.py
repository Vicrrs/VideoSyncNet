import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregando imagem
img = cv2.imread("/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg")
rows, cols, _ = img.shape

# Definindo 3 pontos na imagem original
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# Define os pontos correspondentes na imagem transformada (aplicando rotação, translação e escala)
pts2 = np.float32([[70, 100], [220, 80], [100, 250]])

# Calcula a matriz de transformacao afim
M = cv2.getAffineTransform(pts1, pts2)

# Aplica a transformacao afim
img_trans = cv2.warpAffine(img, M, (cols, rows))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Imagem Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_trans, cv2.COLOR_BGR2RGB))
plt.title("Imagem Transformada")
plt.axis("off")
plt.show()

