import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImagePreprocessor:
    def __init__(self, apply_gaussian=True, apply_bilateral=True, apply_median=True, 
                 apply_clahe=True, apply_perspective_correction=True):
        self.apply_gaussian = apply_gaussian
        self.apply_bilateral = apply_bilateral
        self.apply_median = apply_median
        self.apply_clahe = apply_clahe
        self.apply_perspective_correction = apply_perspective_correction

    def normalize_intensity(self, image):
        """ Normaliza a intensidade dos pixels para a faixa [0, 255] (uint8) """
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image.astype(np.uint8)  # Converte de volta para uint8

    def remove_noise(self, image):
        """ Aplica filtros para remoção de ruído """
        if self.apply_gaussian:
            image = cv2.GaussianBlur(image, (5, 5), 0)
        if self.apply_bilateral:
            image = cv2.bilateralFilter(image, 9, 75, 75)
        if self.apply_median:
            image = cv2.medianBlur(image, 5)
        return image

    def apply_clahe_equalization(self, image):
        """ Aplica a equalização adaptativa CLAHE para melhorar contraste """
        if len(image.shape) == 3 and image.shape[2] == 3:  # Verifica se a imagem é colorida (RGB)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)  # CLAHE aplicado corretamente
            lab = cv2.merge((l, a, b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:  # Imagem em escala de cinza
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
        return image

    def correct_perspective(self, image):
        """ Aplica correção de perspectiva utilizando pontos de referência fixos """
        if not self.apply_perspective_correction:
            return image

        h, w = image.shape[:2]
        src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])  # Pontos originais
        dst_pts = np.float32([[10, 20], [w - 10, 30], [15, h - 20], [w - 20, h - 10]])  # Nova posição

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        corrected_image = cv2.warpPerspective(image, matrix, (w, h))
        return corrected_image

    def preprocess_image(self, image_path):
        """ Executa todo o pipeline de pré-processamento de imagem """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Erro ao carregar a imagem: {image_path}")

        image = self.normalize_intensity(image)  # Agora em uint8 corretamente
        image = self.remove_noise(image)
        image = self.apply_clahe_equalization(image)
        image = self.correct_perspective(image)

        return image

    def display_images(self, original, processed):
        """ Exibe a imagem original e a processada lado a lado """
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Imagem Original")
        ax[0].axis("off")

        ax[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        ax[1].set_title("Imagem Processada")
        ax[1].axis("off")

        plt.show()

# --- Execução ---
if __name__ == "__main__":
    image_path = "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg"
    preprocessor = ImagePreprocessor()

    original_image = cv2.imread(image_path)
    processed_image = preprocessor.preprocess_image(image_path)

    preprocessor.display_images(original_image, processed_image)
