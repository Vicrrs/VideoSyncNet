import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Fluxo Óptico (Lucas-Kanade, Farneback) → para medir deslocamento entre quadros.
Estimativa de deformação temporal → alinha eventos que ocorrem em momentos diferentes.
"""

# Carregar o vídeo
video_path = '/home/vicrrs/projetos/meus_projetos/VideoSyncNet/videos/VR001.MOV'
cap = cv2.VideoCapture(video_path)

ret, frame1 = cap.read()
if not ret:
    print("Erro ao abrir o vídeo")
    cap.release()
    exit()

# Converter o primeiro frame para escala de cinza
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Criar uma imagem HSV (mesmo tamanho do frame) para visualizar o fluxo óptico
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255  # Saturação máxima para melhor visualização

# Ativar o modo interativo do Matplotlib e criar os subplots
plt.ion()
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

while cap.isOpened() and plt.fignum_exists(fig.number):
    ret, frame2 = cap.read()
    if not ret:
        break

    # Converter o frame atual para escala de cinza
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calcular o fluxo óptico usando o método de Farneback
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    # Converter as componentes do fluxo para magnitude e ângulo
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # Mapeia o ângulo para [0,180]
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Converter de BGR para RGB para exibição correta no Matplotlib
    rgb_original = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    rgb_flow = cv2.cvtColor(bgr_flow, cv2.COLOR_BGR2RGB)

    # Atualizar o subplot com o vídeo original
    axs[0].clear()
    axs[0].imshow(rgb_original)
    axs[0].set_title("Vídeo Original")
    axs[0].axis("off")

    # Atualizar o subplot com o fluxo óptico
    axs[1].clear()
    axs[1].imshow(rgb_flow)
    axs[1].set_title("Fluxo Óptico")
    axs[1].axis("off")

    plt.tight_layout()
    plt.pause(0.001)

    # Preparar o próximo frame: o frame atual passa a ser o anterior
    prvs = next_frame

# Encerrar a captura e fechar as janelas do Matplotlib
cap.release()
plt.ioff()
plt.close()
