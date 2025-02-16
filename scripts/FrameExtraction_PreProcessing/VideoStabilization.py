import cv2
import numpy as np

video_path = '/home/vicrrs/projetos/meus_projetos/VideoSyncNet/videos/setup.mp4'
cap = cv2.VideoCapture(video_path)

# Parâmetros para o detector de cantos
feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
# Parâmetros para o método de Lucas-Kanade (fluxo óptico)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Leitura do primeiro quadro
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Preparar para gravação do vídeo estabilizado
height, width = prev_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_estabilizado.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calcula o fluxo óptico para encontrar novos pontos
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    
    # Seleciona pontos válidos
    good_prev = prev_pts[status==1]
    good_curr = curr_pts[status==1]
    
    # Estima a transformação afim (aqui, usamos apenas translação para simplificação)
    if len(good_prev) >= 3:
        M, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr)
        # Aplica a transformação inversa para estabilizar o quadro
        frame_stabilized = cv2.warpAffine(frame, M, (width, height))
    else:
        frame_stabilized = frame.copy()
    
    out.write(frame_stabilized)
    
    # Atualiza para o próximo quadro
    prev_gray = curr_gray.copy()
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

cap.release()
out.release()
print("Vídeo estabilizado gerado com sucesso.")
