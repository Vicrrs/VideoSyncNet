import cv2
import numpy as np

# Exemplo: Extração de características SIFT
img = cv2.imread('/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(img, None)

# Normaliza os descritores para serem usados na rede
descriptors_norm = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)

print("Formato dos descritores:", descriptors_norm.shape)
