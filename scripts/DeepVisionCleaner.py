import cv2
import numpy as np
import matplotlib.pyplot as plt

def unsharp_mask(image, ksize=(5, 5), alpha=1.5, beta=-0.5):
    """
    Aplica Unsharp Mask para dar mais nitidez à imagem.
    alpha > 1 -> realça detalhes
    beta < 0 -> subtrai blur
    """
    blurred = cv2.GaussianBlur(image, ksize, 0)
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, 0)
    return sharpened

def adjust_gamma(image, gamma=1.2):
    """
    Corrige gamma da imagem para realçar áreas escuras ou claras.
    gamma > 1.0 -> clareia
    gamma < 1.0 -> escurece
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

class ImagePreprocessor:
    """
    Versão modificada para intensificar a diferença visual.
    """
    def __init__(self,
                 apply_gaussian=True,
                 apply_bilateral=True,
                 apply_median=True,
                 apply_clahe=True,
                 apply_perspective_correction=True,
                 apply_unsharp=True,
                 apply_gamma=True):
        self.apply_gaussian = apply_gaussian
        self.apply_bilateral = apply_bilateral
        self.apply_median = apply_median
        self.apply_clahe = apply_clahe
        self.apply_perspective_correction = apply_perspective_correction
        self.apply_unsharp = apply_unsharp
        self.apply_gamma = apply_gamma

        # Ajustes de parâmetros
        self.gaussian_ksize = (9, 9)       # kernel maior
        self.bilateral_d = 15             # maior intensidade
        self.bilateral_sigma_color = 150
        self.bilateral_sigma_space = 150
        self.median_ksize = 5
        self.clip_limit = 4.0
        self.tile_grid_size = (4, 4)
        self.gamma_value = 1.2            # realçar ligeiramente

    def normalize_intensity(self, image):
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image.astype(np.uint8)

    def remove_noise(self, image):
        if self.apply_gaussian:
            image = cv2.GaussianBlur(image, self.gaussian_ksize, 0)
        if self.apply_bilateral:
            image = cv2.bilateralFilter(image, 
                                        self.bilateral_d,
                                        self.bilateral_sigma_color,
                                        self.bilateral_sigma_space)
        if self.apply_median:
            image = cv2.medianBlur(image, self.median_ksize)
        return image

    def apply_clahe_equalization(self, image):
        if not self.apply_clahe:
            return image

        if len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            image = clahe.apply(image)
        return image

    def correct_perspective(self, image):
        if not self.apply_perspective_correction:
            return image
        h, w = image.shape[:2]
        src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        dst_pts = np.float32([[10, 20], [w - 10, 30], [15, h - 20], [w - 20, h - 10]])
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        corrected_image = cv2.warpPerspective(image, matrix, (w, h))
        return corrected_image

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Erro ao carregar a imagem: {image_path}")

        # 1) Normalização
        image = self.normalize_intensity(image)
        # 2) Remoção de Ruído
        image = self.remove_noise(image)
        # 3) CLAHE
        image = self.apply_clahe_equalization(image)
        # 4) Ajuste de Gamma (opcional)
        if self.apply_gamma:
            image = adjust_gamma(image, self.gamma_value)
        # 5) Sharpening (Unsharp Mask)
        if self.apply_unsharp:
            image = unsharp_mask(image, ksize=(5,5), alpha=1.8, beta=-0.6)
        # 6) Correção de Perspectiva
        image = self.correct_perspective(image)

        return image

    def display_images(self, original, processed):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        ax[0].set_title("Imagem Original")
        ax[0].axis("off")

        ax[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        ax[1].set_title("Imagem Processada (++Forte)")
        ax[1].axis("off")

        plt.show()

# --- Execução de Exemplo ---
if __name__ == "__main__":
    image_path = "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg"
    # Instancia com as novas opções
    preprocessor = ImagePreprocessor(
        apply_gaussian=True,
        apply_bilateral=True,
        apply_median=True,
        apply_clahe=True,
        apply_perspective_correction=False,
        apply_unsharp=True,
        apply_gamma=True
    )

    original = cv2.imread(image_path)
    processed = preprocessor.preprocess_image(image_path)
    preprocessor.display_images(original, processed)
