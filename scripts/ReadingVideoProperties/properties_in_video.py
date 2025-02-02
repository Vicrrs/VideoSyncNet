import cv2
import time

# Caminho do vídeo
video_path = '/home/vicrrs/projetos/meus_projetos/VideoSyncNet/videos/setup.mp4'

# Abre o vídeo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
else:
    # Leitura das propriedades do vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)  # FPS teórico do vídeo
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS do arquivo: {fps} FPS")
    print(f"Resolução: {largura}x{altura}")
    print(f"Total de Quadros: {total_frames}")

    # Inicializa a contagem de tempo para calcular o FPS em tempo real
    prev_time = time.time()
    real_time_fps = 0

    while True:
        start_time = time.time()  # Início do tempo do frame atual

        ret, frame = cap.read()
        if not ret:
            break  # Sai do loop se não houver mais quadros

        # Obtém o número do quadro atual
        frame_atual = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Calcula o FPS em tempo real
        elapsed_time = time.time() - prev_time
        prev_time = time.time()
        real_time_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0

        # Adiciona as informações na tela
        cv2.putText(frame, f"FPS Atual: {real_time_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Res: {largura}x{altura}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_atual}/{total_frames}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Exibe o vídeo
        cv2.imshow("Video com Informacoes", frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
