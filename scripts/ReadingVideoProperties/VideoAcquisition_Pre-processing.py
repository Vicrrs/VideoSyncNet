import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Tarefas a serem feitas:

    Converter vídeos para um formato padronizado (ex.: .mp4 com FFmpeg).
    Redimensionar os quadros para um tamanho fixo.
    Normalizar brilho e contraste para reduzir diferenças de iluminação entre câmeras.
    Sincronizar os vídeos com base nos metadados (timestamps) ou pelo áudio.

Ferramentas:

    FFmpeg para manipulação de vídeos.
    OpenCV para carregar e processar os quadros.
    Matplotlib pra mostrar o video
"""

# Caminho do vídeo
video_path = "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/videos/setup.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Ativar o modo interativo do matplotlib e criar a figura com dois subplots
plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

try:
    while cap.isOpened() and plt.fignum_exists(fig.number):
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar o frame para 720p (1280x720)
        frame_resized = cv2.resize(frame, (1280, 720))
        
        # Converter para escala de cinza e normalizar a iluminação
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_eq = cv2.equalizeHist(gray)
        
        # Atualizar os subplots
        ax[0].clear()
        ax[1].clear()
        
        # Exibir o frame original (convertido para RGB)
        ax[0].imshow(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Frame Original")
        ax[0].axis('off')
        
        # Exibir o frame processado (em escala de cinza)
        ax[1].imshow(frame_eq, cmap='gray')
        ax[1].set_title("Frame Processado")
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.pause(0.001)
except KeyboardInterrupt:
    print("Interrompido pelo usuário.")
finally:
    # Liberar recursos e fechar janelas
    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close('all')
