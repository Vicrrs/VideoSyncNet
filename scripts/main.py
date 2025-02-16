import cv2
import os
import numpy as np
import torch
import argparse

from DeepVisionCleaner import ImagePreprocessor
from feature_extractor import FeatureExtractor
from spatial_temporal_alignment import SpatialTemporalAligner


def align_videos_incrementally(video1_path, video2_path, output_dir="output_frames_incremental"):
    """
    Alinha dois vídeos de forma incremental (frame a frame).
    """

    # Cria a pasta de saída
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Instancia pipeline
    preprocessor = ImagePreprocessor(
        apply_gaussian=True,
        apply_bilateral=True,
        apply_median=True,
        apply_clahe=True,
        apply_perspective_correction=False
    )
    extractor = FeatureExtractor(
        use_cnn=False,          
        use_hog=True,
        use_optical_flow=False,
        use_siamese=True        
    )
    aligner = SpatialTemporalAligner(
        use_epipolar=False,
        use_warping=False,  
        use_siamese=True
    )

    # Abertura dos dois vídeos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Erro ao abrir um dos vídeos. Verifique o caminho.")
        return

    # Homografia acumulada (começa com identidade)
    H_acc = np.eye(3, dtype=np.float64)
    frame_count = 0

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("Fim de um dos vídeos ou erro de leitura.")
            break

        # --- (1) Pré-processamento ---
        pre1 = preprocessor.normalize_intensity(frame1)
        pre1 = preprocessor.remove_noise(pre1)
        pre1 = preprocessor.apply_clahe_equalization(pre1)

        pre2 = preprocessor.normalize_intensity(frame2)
        pre2 = preprocessor.remove_noise(pre2)
        pre2 = preprocessor.apply_clahe_equalization(pre2)

        h, w = pre1.shape[:2]

        # --- (2) Warp inicial usando H_acc (chute do quadro anterior) ---
        warp2 = cv2.warpPerspective(pre2, H_acc, (w, h))

        # --- (3) Detecta/casa keypoints entre pre1 e warp2 (ORB) ---
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
            good_matches = matches[:50]  # Seleciona os melhores 50

            if len(good_matches) >= 4:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                H_incr_found, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
                if H_incr_found is not None:
                    H_incr = H_incr_found

        # --- (4) Atualiza homografia acumulada ---
        H_acc = H_incr @ H_acc

        # --- (5) Warp final do frame2 original usando H_acc atualizado ---
        aligned_frame2 = cv2.warpPerspective(pre2, H_acc, (w, h))

        # (Opcional) Distância siamese (ResNet18)
        if aligner.use_siamese:
            sim_resnet = aligner.compute_siamese_similarity(pre1, aligned_frame2)
            print(f"[Frame {frame_count}] Dist. ResNet18 (incremental) = {sim_resnet:.4f}")

        # (Opcional) Distância euclidiana simples (gray-level)
        siamese_dist_simple = extractor.compute_siamese_distance(pre1, pre2)

        # --- (6) Monta imagem de debug (4 quadrantes) ---
        combined_top = np.hstack([pre1, pre2])
        combined_bottom = np.hstack([warp2, aligned_frame2])
        display = np.vstack([combined_top, combined_bottom])

        # Info no frame
        text_info = f"Frame #{frame_count} | Dist Simples: {siamese_dist_simple:.2f}"
        cv2.putText(display, text_info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --- (7) Salva resultado ---
        out_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
        cv2.imwrite(out_filename, display)
        print(f"Frame {frame_count} salvo em {out_filename}")

        frame_count += 1

    cap1.release()
    cap2.release()
    print(f"\nConcluído. {frame_count} frames salvos em '{output_dir}'.")


def align_image_sequences_incrementally(folder1, folder2, output_dir="output_frames_incremental"):
    """
    Alinha duas sequências de imagens (cada pasta = 1 "vídeo") de forma incremental.
    Usa a mesma lógica de align_videos_incrementally, mas lendo de pastas.
    """

    # Cria a pasta de saída
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lista de arquivos
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(valid_exts)])
    files2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(valid_exts)])

    if len(files1) == 0 or len(files2) == 0:
        print("Uma das pastas não tem imagens compatíveis. Verifique seus caminhos.")
        return

    # Instancia pipeline
    preprocessor = ImagePreprocessor(
        apply_gaussian=True,
        apply_bilateral=True,
        apply_median=True,
        apply_clahe=True,
        apply_perspective_correction=False
    )
    extractor = FeatureExtractor(
        use_cnn=False,  
        use_hog=True,
        use_optical_flow=False,
        use_siamese=True  
    )
    aligner = SpatialTemporalAligner(
        use_epipolar=False,
        use_warping=False,  
        use_siamese=True
    )

    # Homografia acumulada
    H_acc = np.eye(3, dtype=np.float64)
    frame_count = 0
    min_len = min(len(files1), len(files2))

    for i in range(min_len):
        path1 = os.path.join(folder1, files1[i])
        path2 = os.path.join(folder2, files2[i])

        frame1 = cv2.imread(path1)
        frame2 = cv2.imread(path2)
        if frame1 is None or frame2 is None:
            print(f"Falha ao carregar {path1} ou {path2}. Pulando...")
            continue

        # (1) Pré-processa
        pre1 = preprocessor.normalize_intensity(frame1)
        pre1 = preprocessor.remove_noise(pre1)
        pre1 = preprocessor.apply_clahe_equalization(pre1)

        pre2 = preprocessor.normalize_intensity(frame2)
        pre2 = preprocessor.remove_noise(pre2)
        pre2 = preprocessor.apply_clahe_equalization(pre2)

        # Dimensões
        h, w = pre1.shape[:2]

        # (2) Warp inicial
        warp2 = cv2.warpPerspective(pre2, H_acc, (w, h))

        # (3) ORB
        gray1 = cv2.cvtColor(pre1, cv2.COLOR_BGR2GRAY)
        gray2_warp = cv2.cvtColor(warp2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2_warp, None)

        H_incr = np.eye(3, dtype=np.float64)
        if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches[:50]

            if len(good_matches) >= 4:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                H_incr_found, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
                if H_incr_found is not None:
                    H_incr = H_incr_found

        # (4) Atualiza H_acc
        H_acc = H_incr @ H_acc

        # (5) Warp final
        aligned_frame2 = cv2.warpPerspective(pre2, H_acc, (w, h))

        # Distância siamese
        if aligner.use_siamese:
            sim_resnet = aligner.compute_siamese_similarity(pre1, aligned_frame2)
            print(f"[Frame {frame_count}] Dist. ResNet18 (incremental) = {sim_resnet:.4f}")

        # Dist. euclidiana simples
        siamese_dist_simple = extractor.compute_siamese_distance(pre1, pre2)

        # (6) Monta visualização
        combined_top = np.hstack([pre1, pre2])
        combined_bottom = np.hstack([warp2, aligned_frame2])
        display = np.vstack([combined_top, combined_bottom])

        text_info = f"Frame #{frame_count} | Dist Simples: {siamese_dist_simple:.2f}"
        cv2.putText(display, text_info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # (7) Salva
        out_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.png")
        cv2.imwrite(out_filename, display)
        print(f"Frame {frame_count} salvo em {out_filename}")

        frame_count += 1

    print(f"\nConcluído. {frame_count} frames salvos em '{output_dir}'.")


def main():
    """
    Exemplo de CLI (Command-Line Interface) para escolher entre alinhar
    dois vídeos ou duas pastas de imagens.
    
    Exemplo de uso:
        python main_alignment.py --mode video  --input1 video1.mp4 --input2 video2.mp4 --output_dir saida_videos
        python main_alignment.py --mode images --input1 pasta1    --input2 pasta2    --output_dir saida_imgs
    """
    parser = argparse.ArgumentParser(description="Alinhamento incremental de vídeos ou sequências de imagens.")
    parser.add_argument("--mode", choices=["video","images"], default="images",
                        help="Escolha se o input será 'video' ou 'images'.")
    parser.add_argument("--input1", type=str, required=True, help="Caminho do vídeo 1 ou pasta de imagens 1.")
    parser.add_argument("--input2", type=str, required=True, help="Caminho do vídeo 2 ou pasta de imagens 2.")
    parser.add_argument("--output_dir", type=str, default="output_frames_incremental", 
                        help="Pasta para salvar os frames de saída.")
    args = parser.parse_args()

    if args.mode == "video":
        align_videos_incrementally(args.input1, args.input2, args.output_dir)
    else:
        align_image_sequences_incrementally(args.input1, args.input2, args.output_dir)


if __name__ == "__main__":
    main()
