import cv2
import numpy as np

def gamma_correction(image, gamma=2.2):
    invGamma = 1.0 / gamma

    # Criando uma tabela de lookup pra acelerar o processamento
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Carrega um quadro de exemplo
frame = cv2.imread('/home/vicrrs/projetos/meus_projetos/VideoSyncNet/scripts/FrameExtraction_PreProcessing/frame_0000.png')
frame_corr = gamma_correction(frame, gamma=2.2)

cv2.imshow('Original', frame)
cv2.imshow('Gamma Corrigido', frame_corr)
cv2.waitKey(0)
cv2.destroyAllWindows()
