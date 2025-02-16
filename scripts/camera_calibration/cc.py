import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configurações do tabuleiro de xadrez
chessboard_size = (9, 6)  # Confirme se é o número correto de interseções
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Lista de imagens
image_paths = [
    "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/chessboard_images/chess001.jpg",
    "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/chessboard_images/chess002.jpg",
    "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/chessboard_images/chess003.jpg",
    "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/chessboard_images/chess004.jpg",
    "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/chessboard_images/chess005.jpg",
    "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/chessboard_images/chess006.jpg",
    "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/chessboard_images/chess007.jpg"
]

for fname in image_paths:
    img = cv2.imread(fname)
    if img is None:
        print(f"Erro ao carregar a imagem: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Exibir imagem em escala de cinza para verificar contraste
    plt.imshow(gray, cmap='gray')
    plt.title(f"Imagem Convertida para Cinza - {fname}")
    plt.show()

    # Aplicar detecção de bordas
    edges = cv2.Canny(gray, 50, 150)
    plt.imshow(edges, cmap='gray')
    plt.title("Detecção de Bordas")
    plt.show()

    # Encontrar os cantos do tabuleiro
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, 
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                             cv2.CALIB_CB_NORMALIZE_IMAGE +
                                             cv2.CALIB_CB_FAST_CHECK)

    if ret:
        print(f"Tabuleiro detectado na imagem {fname}")
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Tabuleiro Detectado")
        plt.show()
    else:
        print(f"NÃO foi possível detectar o tabuleiro em {fname}")
