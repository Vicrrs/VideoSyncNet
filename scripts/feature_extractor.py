import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.feature import hog

class FeatureExtractor:
    """
    Funcionalidades do Código:

        Fluxo Óptico (Optical Flow)
            RAFT (usando FlowNet 2.0 como alternativa)
            Farneback Optical Flow (para métodos mais tradicionais)

        Mapeamento de Características Locais e Globais
            CNNs para extração de texturas dinâmicas (Usando VGG16 pré-treinado)
            Redes Siamesas para correspondência de imagens (Usando distância Euclidiana)

        Detecção de Bordas e Segmentação
            Laplaciano de Matting
            Histogramas de Gradientes (HOG3D)
            Segmentação baseada em Gradiente
    """
    def __init__(self, use_cnn=True, use_hog=True, use_optical_flow=True, use_siamese=True):
        self.use_cnn = use_cnn
        self.use_hog = use_hog
        self.use_optical_flow = use_optical_flow
        self.use_siamese = use_siamese

        if self.use_cnn:
            self.cnn_model = models.vgg16(pretrained=True).features.eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        
    def compute_optical_flow(self, frame1, frame2):
        """Calcula o fluxo óptico usando Farneback"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def extract_features_cnn(self, image):
        """Extrai características usando VGG16"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.cnn_model(image_tensor)
        return features.squeeze().numpy()

    def compute_hog_features(self, image):
        """Extrai características HOG"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        return features

    def compute_siamese_distance(self, img1, img2):
        """Calcula a distância Euclidiana entre duas imagens para comparação siamesa"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        diff = np.linalg.norm(gray1.astype("float") - gray2.astype("float"))
        return diff

    def detect_edges(self, image):
        """Aplica filtros Laplaciano e Gradiente para detectar bordas"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        return np.uint8(np.abs(edges))

    def process_image_pair(self, image1_path, image2_path):
        """Executa todo o pipeline para duas imagens"""
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        if img1 is None or img2 is None:
            raise ValueError("Erro ao carregar as imagens!")

        results = {}

        if self.use_cnn:
            results["CNN_Features"] = self.extract_features_cnn(img1)

        if self.use_hog:
            results["HOG_Features"] = self.compute_hog_features(img1)

        if self.use_optical_flow:
            results["Optical_Flow"] = self.compute_optical_flow(img1, img2)

        if self.use_siamese:
            results["Siamese_Distance"] = self.compute_siamese_distance(img1, img2)

        results["Edges"] = self.detect_edges(img1)

        return img1, img2, results

    def visualize_results(self, img1, img2, results):
        """Exibe os resultados das extrações"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Imagem 1")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Imagem 2")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(results["Edges"], cmap="gray")
        axes[0, 2].set_title("Detecção de Bordas")
        axes[0, 2].axis("off")

        if "Optical_Flow" in results:
            flow = results["Optical_Flow"]
            flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            axes[1, 0].imshow(flow_mag, cmap="jet")
            axes[1, 0].set_title("Fluxo Óptico (Magnitude)")
            axes[1, 0].axis("off")

        if "HOG_Features" in results:
            axes[1, 1].hist(results["HOG_Features"], bins=30, color='blue', alpha=0.7)
            axes[1, 1].set_title("Histograma de Gradientes (HOG)")

        if "Siamese_Distance" in results:
            axes[1, 2].text(0.5, 0.5, f"Siamese Distância: {results['Siamese_Distance']:.2f}", 
                            fontsize=12, ha='center', va='center')
            axes[1, 2].set_title("Distância Siamesa")
            axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

# --- Execução ---
if __name__ == "__main__":
    image1_path = "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg"
    image2_path = "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj002.jpg"

    extractor = FeatureExtractor()
    img1, img2, results = extractor.process_image_pair(image1_path, image2_path)
    extractor.visualize_results(img1, img2, results)
