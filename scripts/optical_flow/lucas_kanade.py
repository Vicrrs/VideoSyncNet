import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega duas imagens consecutivas em escala de cinza
img1 = cv2.imread('/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj002.jpg', cv2.IMREAD_GRAYSCALE)

# Parâmetros para o detector de cantos (Shi-Tomasi)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(img1, mask=None, **feature_params)

# Parâmetros para o método de Lucas-Kanade
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Calcula o fluxo óptico
p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

# Seleciona os bons pontos
good_new = p1[st==1]
good_old = p0[st==1]

# Visualização dos vetores de fluxo
img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    cv2.arrowedLine(img2_color, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
plt.title("Fluxo Óptico com Lucas-Kanade")
plt.axis("off")
plt.show()
