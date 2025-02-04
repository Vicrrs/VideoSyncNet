import cv2
import matplotlib.pyplot as plt

"""
O SIFT extrai pontos de interesse invariante a escala e rotação
"""

# Carregando as imagens
img1 = cv2.imread("/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj002.jpg", cv2.IMREAD_GRAYSCALE)

# Inicializando detector SIFT
sift = cv2.SIFT_create()

# Detectando pontos-chave e computa os descritores
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Matcher de forca bruta com a norma L2
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Ordena os matches pela distancia
matches = sorted(matches, key=lambda x: x.distance)

# Desenha os primeiros 20 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.title("Correspondência de Características com SIFT")
plt.axis("off")
plt.show()
