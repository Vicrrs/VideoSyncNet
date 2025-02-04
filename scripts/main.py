import cv2
import os
import numpy as np
import torch

from DeepVisionCleaner import ImagePreprocessor
from feature_extractor import FeatureExtractor
from spatial_temporal_alignment import SpatialTemporalAligner

def align_videos_incrementally(video1_path, video2_path, output_dir="output_frames_incremental"):
    """
    Alinha dois vídeos de forma incremental:
      - Lê frame a frame de cada vídeo.
      - Pré-processa (remoção de ruído, CLAHE, etc.).
      - Mantém uma homografia acumulada (H_acc).
      - Para cada novo par de frames:
          1) Warp do frame2 com H_acc (chute inicial).
          2) Detecta/casa keypoints (ORB) entre frame1 e esse warp2.
          3) Estima homografia incremental (H_incr) e compõe H_acc = H_incr * H_acc.
          4) Aplica H_acc ao frame2 original para gerar aligned_frame2 final.
          5) (Opcional) Calcula distância siamese com ResNet18.
          6) Salva a imagem de saída em output_dir.

    Observação:
      - Em cenas muito 3D, podem ocorrer distorções ou falhas de alinhamento,
        pois a homografia modela um plano. Mas se a mudança de perspectiva for pequena,
        esse método incremental tende a ser mais estável do que recalcular tudo a cada frame.
    """

    # Cria a pasta de saída, se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Inicializa pipeline
    preprocessor = ImagePreprocessor(
        apply_gaussian=True,
        apply_bilateral=True,
        apply_median=True,
        apply_clahe=True,
        apply_perspective_correction=False  # Se True, aplica perspectiva fixa do preprocessor
    )
    extractor = FeatureExtractor(
        use_cnn=True,          # True se quiser extrair CNN (VGG16) em cada frame
        use_hog=True,
        use_optical_flow=True,
        use_siamese=True        # Distância euclidiana simples (gray-level)
    )
    # Usaremos a classe para a distância ResNet18 (Siamese) mas
    # NÃO usaremos o warp dela, pois faremos warp "na mão" no loop.
    aligner = SpatialTemporalAligner(
        use_epipolar=False,
        use_warping=False,   # Desativado, pois a homografia incremental será calculada aqui
        use_siamese=True
    )

    # Abertura dos dois vídeos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Erro ao abrir um dos vídeos. Verifique o caminho.")
        return

    # Homografia acumulada (começa com a identidade)
    H_acc = np.eye(3, dtype=np.float64)

    frame_count = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("Fim de um dos vídeos ou erro de leitura.")
            break

        # --------------------------
        # 1) Pré-processamento
        # --------------------------
        pre1 = preprocessor.normalize_intensity(frame1)
        pre1 = preprocessor.remove_noise(pre1)
        pre1 = preprocessor.apply_clahe_equalization(pre1)
        # pre1 = preprocessor.correct_perspective(pre1)  # se quiser

        pre2 = preprocessor.normalize_intensity(frame2)
        pre2 = preprocessor.remove_noise(pre2)
        pre2 = preprocessor.apply_clahe_equalization(pre2)
        # pre2 = preprocessor.correct_perspective(pre2)

        # --------------------------
        # 2) Extração de características simples (opcional)
        # --------------------------
        hog1 = extractor.compute_hog_features(pre1)
        hog2 = extractor.compute_hog_features(pre2)
        siamese_dist_simple = extractor.compute_siamese_distance(pre1, pre2)

        # Dimensões para warp
        h, w = pre1.shape[:2]

        # --------------------------
        # 3) Warp inicial usando H_acc (chute do quadro anterior)
        # --------------------------
        warp2 = cv2.warpPerspective(pre2, H_acc, (w, h))  

        # --------------------------
        # 4) Detecta/casa keypoints (ORB) entre pre1 e warp2
        # --------------------------
        gray1 = cv2.cvtColor(pre1, cv2.COLOR_BGR2GRAY)
        gray2_warp = cv2.cvtColor(warp2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2_warp, None)

        # Homografia incremental (default = identidade)
        H_incr = np.eye(3, dtype=np.float64)
        if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:50]  # pega os melhores 50

            if len(good_matches) >= 4:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Estima homografia incremental (RANSAC)
                H_incr_found, mask_h = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
                if H_incr_found is not None:
                    H_incr = H_incr_found

        # --------------------------
        # 5) Atualiza homografia acumulada
        # --------------------------
        # Multiplicamos a homografia incremental pela acumulada
        # (ordem: H_incr atua primeiro, depois H_acc anterior)
        H_acc = H_incr @ H_acc

        # --------------------------
        # 6) Warp final do pre2 original usando H_acc atualizado
        # --------------------------
        aligned_frame2 = cv2.warpPerspective(pre2, H_acc, (w, h))

        # (Opcional) Distância siamese com ResNet18
        if aligner.use_siamese:
            sim_resnet = aligner.compute_siamese_similarity(pre1, aligned_frame2)
            print(f"[Frame {frame_count}] Dist. ResNet18 (incremental) = {sim_resnet:.4f}")

        # --------------------------
        # 7) Monta imagem de saída (4 quadrantes)
        # --------------------------
        #  - Superior Esq = pre1 (referência)
        #  - Superior Dir = pre2 (sem warp)
        #  - Inferior Esq = warp2 (antes do refinamento incremental)
        #  - Inferior Dir = aligned_frame2 (após refinamento)
        combined_top = np.hstack([pre1, pre2])
        combined_bottom = np.hstack([warp2, aligned_frame2])
        display = np.vstack([combined_top, combined_bottom])

        # Info no frame
        text_info = f"Frame #{frame_count} | Dist Simples: {siamese_dist_simple:.2f}"
        cv2.putText(display, text_info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --------------------------
        # 8) Salva no disco
        # --------------------------
        out_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
        cv2.imwrite(out_filename, display)

        print(f"Frame {frame_count} salvo em {out_filename}")
        frame_count += 1

    # Final
    cap1.release()
    cap2.release()
    print(f"Concluído. {frame_count} frames salvos em '{output_dir}'.")


if __name__ == "__main__":
    video1 = "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/videos/Teste01/Matheus.mov"
    video2 = "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/videos/Teste01/Vinicius.mov"

    align_videos_incrementally(video1, video2)
