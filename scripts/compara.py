import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Usa um backend que não depende de interface gráfica
import matplotlib.pyplot as plt
from DeepVisionCleaner import ImagePreprocessor
from feature_extractor import FeatureExtractor

# Caminho da pasta de teste (ex: 'imgs/garrafa')
test_folder = "/home/vicrrs/projetos/meus_projetos/VideoSyncNet/imgs/garrafa"

# Lista de arquivos de imagem válidos, ordenados
valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
image_files = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(valid_exts)])

# Instancia os módulos de pré-processamento e extração
preprocessor = ImagePreprocessor(apply_perspective_correction=False)
extractor = FeatureExtractor(use_siamese=True)

# Listas para armazenar as distâncias (por pares consecutivos)
distances_raw = []
distances_processed = []
frame_indices = []

# Percorre os pares consecutivos
for i in range(len(image_files) - 1):
    # Monta o caminho completo das imagens
    img1_path = os.path.join(test_folder, image_files[i])
    img2_path = os.path.join(test_folder, image_files[i + 1])
    
    # Carrega as imagens originais (sem tratamento)
    img1_raw = cv2.imread(img1_path)
    img2_raw = cv2.imread(img2_path)
    
    # Se alguma imagem não for carregada, pula o par
    if img1_raw is None or img2_raw is None:
        continue
    
    # Aplica o pré-processamento nas imagens
    img1_proc = preprocessor.preprocess_image(img1_path)
    img2_proc = preprocessor.preprocess_image(img2_path)
    
    # Calcula a distância siamesa entre os pares
    dist_raw = extractor.compute_siamese_distance(img1_raw, img2_raw)
    dist_proc = extractor.compute_siamese_distance(img1_proc, img2_proc)
    
    distances_raw.append(dist_raw)
    distances_processed.append(dist_proc)
    frame_indices.append(i)

    print(f"Par {i}-{i+1} | Raw: {dist_raw:.2f} | Processado: {dist_proc:.2f}")

# Gera um gráfico comparativo
plt.figure(figsize=(8, 5))
plt.plot(frame_indices, distances_raw, marker='o', label='Sem Tratamento')
plt.plot(frame_indices, distances_processed, marker='o', label='Com Tratamento')
plt.xlabel("Índice do Par de Imagens")
plt.ylabel("Distância Siamesa")
plt.title("Comparação de Alinhamento Espaço-Temporal\nSem vs Com Tratamento de Imagem")
plt.legend()
plt.grid(True)

# Salva o gráfico em um arquivo, já que o backend 'Agg' não mostra a janela
plt.savefig("comparacao_alinhamento.png")
print("Gráfico salvo em 'comparacao_alinhamento.png'")
