import cv2
import matplotlib.pyplot as plt

"""
O detector ORB extrai os pontos de interesse e seus descritores das imagens.
Em seguida, utiliza-se o Brute-Force Matcher com a métrica de Hamming para encontrar correspondências.
"""

# Carregando as imagens
img1 = cv2.imread("/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj002.jpg", cv2.IMREAD_GRAYSCALE)

# Inicializa o detector ORB
orb = cv2.ORB_create()

# Detecta e computa os descritores
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Matcher para correspondência de descritores
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Ordena os matches pela distancia
matches = sorted(matches, key=lambda x: x.distance)

# Desenha os primeiros 20 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.title("Correspondência de Características com ORB")
plt.axis("off")
plt.show()
