import cv2
import numpy as np
import matplotlib.pyplot as plt

# Abrir o vídeo
cap = cv2.VideoCapture('/home/vicrrs/projetos/meus_projetos/VideoSyncNet/videos/VR001.MOV')
ret, prev_frame = cap.read()
if not ret:
    print("Erro ao abrir o vídeo.")
    cap.release()
    exit()

# Converter o primeiro frame para escala de cinza
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Configurar o Matplotlib em modo interativo com dois subplots (lado a lado)
plt.ion()
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

while cap.isOpened() and plt.fignum_exists(fig.number):
    ret, frame = cap.read()
    if not ret:
        break

    # Converter o frame atual para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular o fluxo óptico com o método Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    # Criar uma imagem HSV para visualizar o fluxo óptico
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255  # Saturação máxima

    # Converter as componentes do fluxo para magnitude e ângulo
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # Mapeia o ângulo para o intervalo [0, 180]
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Converter a imagem HSV para BGR (para visualização)
    flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Converter as imagens de BGR para RGB para exibição correta no Matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    flow_rgb = cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB)

    # Atualizar o subplot com o frame original
    axs[0].clear()
    axs[0].imshow(frame_rgb)
    axs[0].set_title("Vídeo Original")
    axs[0].axis("off")

    # Atualizar o subplot com o fluxo óptico
    axs[1].clear()
    axs[1].imshow(flow_rgb)
    axs[1].set_title("Fluxo Óptico (Farneback)")
    axs[1].axis("off")

    plt.tight_layout()
    plt.pause(0.001)

    # Atualizar o frame anterior para o próximo cálculo
    prev_gray = gray

# Encerrar a captura e fechar o Matplotlib
cap.release()
plt.ioff()
plt.close()
