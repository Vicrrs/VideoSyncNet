import cv2
import numpy as np
import matplotlib.pyplot as plt

def align_images(img_ref, img_to_align, max_features=5000, good_match_percent=0.15):
    """
    Alinha img_to_align para que ela tenha a mesma perspectiva de img_ref.

    Parâmetros:
        img_ref: imagem de referência (base de alinhamento).
        img_to_align: imagem que será alinhada.
        max_features: número máximo de pontos-chave a serem detectados.
        good_match_percent: percentual de melhores matches a serem considerados.
    
    Retorna:
        aligned_img: imagem alinhada.
        H: matriz de homografia estimada.
        matches: correspondências encontradas.
        keypoints_ref: pontos-chave da imagem de referência.
        keypoints_align: pontos-chave da imagem a ser alinhada.
    """
    # Converter imagens para escala de cinza
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_align = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)
    
    # Detectar pontos-chave e descritores usando ORB
    orb = cv2.ORB_create(max_features)
    keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_ref, None)
    keypoints_align, descriptors_align = orb.detectAndCompute(gray_align, None)
    
    # Fazer correspondência entre os descritores usando o BFMatcher com norma Hamming
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.match(descriptors_ref, descriptors_align, None)
    
    # Ordenar os matches com base na distância (menor distância = melhor correspondência)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Selecionar apenas uma fração dos melhores matches
    num_good_matches = int(len(matches) * good_match_percent)
    matches = matches[:num_good_matches]
    
    # Extrair as localizações dos pontos correspondentes em cada imagem
    pts_ref = np.zeros((len(matches), 2), dtype=np.float32)
    pts_align = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        pts_ref[i, :] = keypoints_ref[match.queryIdx].pt
        pts_align[i, :] = keypoints_align[match.trainIdx].pt
    
    # Estimar a homografia entre as imagens usando RANSAC
    H, mask = cv2.findHomography(pts_align, pts_ref, cv2.RANSAC)
    
    # Aplicar a transformação de homografia para alinhar a imagem
    height, width, channels = img_ref.shape
    aligned_img = cv2.warpPerspective(img_to_align, H, (width, height))
    
    return aligned_img, H, matches, keypoints_ref, keypoints_align

# Exemplo de uso com imagens
if __name__ == "__main__":
    # Carregar imagens de exemplo (substitua os caminhos conforme necessário)
    img_ref = cv2.imread("/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj001.jpg")
    img_to_align = cv2.imread("/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/sj002.jpg")
    
    if img_ref is None or img_to_align is None:
        print("Erro ao carregar as imagens.")
        exit()

    aligned_img, H, matches, kp_ref, kp_align = align_images(img_ref, img_to_align)
    
    # Converter imagens de BGR para RGB para exibir com matplotlib
    img_ref_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    aligned_img_rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)
    
    # Exibir as imagens utilizando matplotlib
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_ref_rgb)
    plt.title("Imagem de Referência")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(aligned_img_rgb)
    plt.title("Imagem Alinhada")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
