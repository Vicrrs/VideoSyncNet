import cv2
import numpy as np

def align_frame(frame_ref, frame_to_align, max_features=5000, good_match_percent=0.15):
    """
    Alinha frame_to_align para que ele tenha a mesma perspectiva de frame_ref.
    Esta função utiliza a mesma lógica do exemplo de imagens.
    """
    gray_ref = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2GRAY)
    gray_align = cv2.cvtColor(frame_to_align, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create(max_features)
    keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_ref, None)
    keypoints_align, descriptors_align = orb.detectAndCompute(gray_align, None)
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(descriptors_ref, descriptors_align, None)
    matches = sorted(matches, key=lambda x: x.distance)
    
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]
    
    pts_ref = np.zeros((len(matches), 2), dtype=np.float32)
    pts_align = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        pts_ref[i, :] = keypoints_ref[match.queryIdx].pt
        pts_align[i, :] = keypoints_align[match.trainIdx].pt

    # Verificar se há matches suficientes para estimar a homografia
    if len(pts_ref) < 4:
        return frame_to_align, None  # Não é possível alinhar com poucos pontos
    
    H, mask = cv2.findHomography(pts_align, pts_ref, cv2.RANSAC)
    height, width, channels = frame_ref.shape
    aligned_frame = cv2.warpPerspective(frame_to_align, H, (width, height))
    
    return aligned_frame, H

def process_videos(video_path_ref, video_path_to_align):
    """
    Processa dois vídeos: um vídeo de referência e um vídeo a ser alinhado.
    """
    cap_ref = cv2.VideoCapture(video_path_ref)
    cap_align = cv2.VideoCapture(video_path_to_align)
    
    if not cap_ref.isOpened() or not cap_align.isOpened():
        print("Erro ao abrir um dos vídeos.")
        return
    
    while True:
        ret_ref, frame_ref = cap_ref.read()
        ret_align, frame_to_align = cap_align.read()
        
        # Se algum vídeo terminar, interrompe o loop
        if not ret_ref or not ret_align:
            break
        
        aligned_frame, H = align_frame(frame_ref, frame_to_align)
        
        # Exibir os frames
        cv2.imshow("Frame de Referencia", frame_ref)
        cv2.imshow("Frame Alinhado", aligned_frame)
        
        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap_ref.release()
    cap_align.release()
    cv2.destroyAllWindows()

# Exemplo de execução para vídeos
if __name__ == "__main__":
    video_ref = "video_referencia.mp4"
    video_to_align = "video_para_alinhar.mp4"
    
    process_videos(video_ref, video_to_align)
