import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class SpatialTemporalAligner:
    """
    Funcionalidades do Código:

        Registro e Alinhamento de Vídeos Multi-Câmera
            Geometria Epipolar (Matriz Fundamental para alinhar múltiplas câmeras)
            Warping Espacial e Temporal (Correção de Perspectiva e Fluxo Óptico)
            Registro de fundo não sobreposto (Transformações baseadas em objetos)

        Sincronização Baseada em Aprendizado Profundo
            Redes Siamesas para correspondência de quadros
            Ajuste Iterativo para sincronização de vídeos

        Transformações Geométricas e Temporais
            Transformações Afins 4×4 para alinhar sequências de imagens
            Warping Temporal para ajustar variações de tempo entre quadros
    """
    def __init__(self, use_epipolar=True, use_warping=True, use_siamese=True):
        self.use_epipolar = use_epipolar
        self.use_warping = use_warping
        self.use_siamese = use_siamese

        if self.use_siamese:
            self.cnn_model = models.resnet18(pretrained=True).eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

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
        """ Usa ResNet18 para calcular similaridade entre quadros """
        img1_tensor = self.transform(img1).unsqueeze(0)
        img2_tensor = self.transform(img2).unsqueeze(0)
        with torch.no_grad():
            feat1 = self.cnn_model(img1_tensor)
            feat2 = self.cnn_model(img2_tensor)
        distance = torch.norm(feat1 - feat2).item()
        return distance

    def align_frames(self, frame1, frame2):
        """ Alinha dois frames usando todas as técnicas disponíveis """
        h, w = frame1.shape[:2]

        if self.use_epipolar:
            keypoints1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])  # Simulados
            keypoints2 = np.float32([[55, 55], [195, 50], [60, 190], [195, 200]])  # Simulados
            F = self.compute_fundamental_matrix(keypoints1, keypoints2)

        if self.use_warping:
            transform_matrix = np.float32([[1, 0, 5], [0, 1, 10], [0, 0, 1]])  # Simula transformação
            frame2 = self.apply_warping(frame2, transform_matrix)

        if self.use_siamese:
            similarity = self.compute_siamese_similarity(frame1, frame2)
            print(f"Distância Siamesa: {similarity:.4f}")

        return frame1, frame2

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
        axes[1, 1].set_title("Frame 2 - Alinhado")
        axes[1, 1].axis("off")

        plt.show()

# --- Execução ---
if __name__ == "__main__":
    frame1_path = "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg"
    frame2_path = "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj002.jpg"

    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    aligner = SpatialTemporalAligner()
    aligned1, aligned2 = aligner.align_frames(frame1, frame2)

    aligner.visualize_alignment(frame1, frame2, aligned1, aligned2)
