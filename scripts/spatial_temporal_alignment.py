import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class SpatialTemporalAligner:
    """
    Funcionalidades do Código:

        - Registro e Alinhamento de Vídeos Multi-Câmera
          (aqui usando ORB + Homografia para warp)
        - Sincronização Baseada em Aprendizado Profundo (opcional)
          (usa ResNet18 para comparação de quadros)
        - Fluxo Óptico e Geometria Epipolar (parcialmente ilustrado)
    """
    def __init__(self, use_epipolar=False, use_warping=True, use_siamese=True):
        self.use_epipolar = use_epipolar
        self.use_warping = use_warping
        self.use_siamese = use_siamese

        # Se for usar siamese, carrega ResNet18 pré-treinada
        if self.use_siamese:
            self.cnn_model = models.resnet18(pretrained=True).eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    # (Métodos opcionais, mantidos para referência)
    def compute_fundamental_matrix(self, points1, points2):
        """ Calcula a matriz fundamental para alinhamento baseado em geometria epipolar """
        fundamental_matrix, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
        return fundamental_matrix

    def apply_warping(self, image, transform_matrix):
        """ Aplica transformação afim ou perspectiva para alinhamento espacial """
        h, w = image.shape[:2]
        warped_image = cv2.warpPerspective(image, transform_matrix, (w, h))
        return warped_image

    def compute_optical_flow(self, frame1, frame2):
        """ Calcula fluxo óptico entre dois quadros """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def compute_siamese_similarity(self, img1, img2):
        """ Usa ResNet18 para calcular similaridade entre quadros (distância) """
        img1_tensor = self.transform(img1).unsqueeze(0)
        img2_tensor = self.transform(img2).unsqueeze(0)
        with torch.no_grad():
            feat1 = self.cnn_model(img1_tensor)
            feat2 = self.cnn_model(img2_tensor)
        distance = torch.norm(feat1 - feat2).item()
        return distance

    def align_frames(self, frame1, frame2):
        """
        Alinha frame2 em relação a frame1 usando:
          - ORB + BFMatcher para encontrar correspondências.
          - Homografia via findHomography (RANSAC).
          - (Opcional) Similaridade siamese (ResNet18).

        Retorna: (frame1, aligned_frame2)
        """
        # Se quisermos epipolar, poderíamos usar findFundamentalMat, mas aqui
        # vamos focar na homografia para realmente alinhar as imagens.

        # Verifica se vamos realmente warp
        if not self.use_warping:
            # Se não for warp, apenas retorna frames inalterados
            if self.use_siamese:
                # Calcula similaridade só para relatório
                sim = self.compute_siamese_similarity(frame1, frame2)
                print(f"Distância Siamesa (sem warp): {sim:.4f}")
            return frame1, frame2

        # 1) Converte para escala de cinza
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 2) Detecta e descreve keypoints com ORB
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            print("Poucos keypoints encontrados. Retornando imagens originais.")
            return frame1, frame2

        # 3) Faz o matching (Brute Force, Hamming para ORB)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        # Ordena matches pelo menor distance
        matches = sorted(matches, key=lambda x: x.distance)

        # 4) Seleciona um subconjunto de bons matches (ex: 50)
        good_matches = matches[:50]
        if len(good_matches) < 4:
            print("Poucos matches para estimar homografia. Retornando imagens originais.")
            return frame1, frame2

        # 5) Extrai as coordenadas (x,y) dos keypoints correspondentes
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 6) Estima a homografia para mapear frame2->frame1
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        if H is None:
            print("Não foi possível calcular homografia. Retornando frames originais.")
            return frame1, frame2

        # 7) Aplica warpPerspective em frame2
        h, w = frame1.shape[:2]
        aligned_frame2 = cv2.warpPerspective(frame2, H, (w, h))

        # 8) Se usar siamese, calcula similaridade (ResNet18)
        if self.use_siamese:
            sim = self.compute_siamese_similarity(frame1, aligned_frame2)
            print(f"Distância Siamesa (ResNet18) após warp: {sim:.4f}")

        return frame1, aligned_frame2

    def visualize_alignment(self, original1, original2, aligned1, aligned2):
        """ Exibe imagens antes e depois do alinhamento """
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].imshow(cv2.cvtColor(original1, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Frame 1 - Original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(original2, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Frame 2 - Original")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(cv2.cvtColor(aligned1, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Frame 1 - Alinhado")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(cv2.cvtColor(aligned2, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Frame 2 - Alinhado (warp)")
        axes[1, 1].axis("off")

        plt.tight_layout()
        plt.show()

# --- Execução para teste ---
if __name__ == "__main__":
    frame1_path = "imagem1.jpg"  # Ajuste para seu caminho real
    frame2_path = "imagem2.jpg"

    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    aligner = SpatialTemporalAligner(
        use_epipolar=False,
        use_warping=True,    # habilita ORB+Homografia
        use_siamese=True
    )
    aligned1, aligned2 = aligner.align_frames(frame1, frame2)

    # Visualiza
    aligner.visualize_alignment(frame1, frame2, aligned1, aligned2)
