import cv2
import matplotlib.pyplot as plt

# Carrega as imagens
img1 = cv2.imread("/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj002.jpg", cv2.IMREAD_GRAYSCALE)

# Inicializa o detector SURF (Necessita OpenCV-contrib)
surf = cv2.xfeatures2d.SURF_create(400)

# Detecta pontos-chave e computa descritores
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

# Matcher de força bruta com norma L2
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# Ordena os matches pela distância
matches = sorted(matches, key=lambda x: x.distance)

# Desenha os primeiros 20 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.title("Correspondência de Características com SURF")
plt.axis("off")
plt.show()
