import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega duas imagens consecutivas em escala de cinza
img1 = cv2.imread('/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj002.jpg', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Erro ao carregar as imagens. Verifique os caminhos!")
    exit()

# Converte as imagens para float64
img1 = img1.astype(np.float64)
img2 = img2.astype(np.float64)

# Calcula os gradientes espaciais (Ix, Iy) e temporais (It)
Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
It = img2 - img1

# Inicializa os campos de fluxo (u: eixo x, v: eixo y)
u = np.zeros(img1.shape)
v = np.zeros(img1.shape)

# Parâmetros do algoritmo
alpha = 1.0  # Regularização para suavidade
n_iter = 100  # Número de iterações

# Itera para calcular os campos de fluxo
for _ in range(n_iter):
    u_avg = cv2.blur(u, (3, 3))
    v_avg = cv2.blur(v, (3, 3))
    
    # Atualização dos vetores de fluxo
    der = Ix * u_avg + Iy * v_avg + It
    denom = alpha**2 + Ix**2 + Iy**2
    u = u_avg - (Ix * der) / denom
    v = v_avg - (Iy * der) / denom

# Reduz a densidade de vetores para visualização
step = 15
y, x = np.mgrid[step//2:img1.shape[0]:step, step//2:img1.shape[1]:step]

# Converte a imagem para BGR para sobreposição dos vetores
img2_color = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Desenha os vetores do fluxo óptico
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        start_x, start_y = int(x[i, j]), int(y[i, j])
        end_x = int(start_x + u[start_y, start_x] * 5)  # Ajuste para melhor visibilidade
        end_y = int(start_y + v[start_y, start_x] * 5)
        
        # Desenha setas na imagem final
        cv2.arrowedLine(img2_color, (start_x, start_y), (end_x, end_y), (0, 255, 0), 1, tipLength=0.4)

# Exibe a imagem com os vetores
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB))
plt.title("Fluxo Óptico com Horn-Schunck")
plt.axis("off")
plt.show()
