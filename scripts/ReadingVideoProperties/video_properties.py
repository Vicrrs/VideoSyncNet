import cv2

# abre o video
video_path = '/home/vicrrs/projetos/meus_projetos/VideoSyncNet/videos/setup.mp4'
cap = cv2.VideoCapture (video_path)

if not cap.isOpened():
    print("Erro ao abrir o video!")
else:
    # Leitura das propriedades
    fps = cap.get(cv2.CAP_PROP_FPS)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Taxa de Quadros: {fps} FPS")
    print(f"Resolução: {largura}x{altura}")
    print(f"Total de Quadros: {total_frames}")

cap.release()
