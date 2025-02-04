import cv2

video_path = '/home/vicrrs/projetos/meus_projetos/VideoSyncNet/videos/setup.mp4'
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Salva o quadro como imagem
    cv2.imwrite(f'frame_{frame_count:04d}.png', frame)
    frame_count += 1

cap.release()
print(f"{frame_count} quadros extraídos.")
