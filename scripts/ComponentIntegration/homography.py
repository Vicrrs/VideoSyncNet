import numpy as np
import cv2

# CNN forneça pontos de interesse
pontos_cnn = np.array([[100, 150], [200, 250], [300, 350]], dtype=np.float32)

# Matriz de homografia (exemplo)
H = np.array([[1.0, 0.0, 10],
              [0.0, 1.0, 20],
              [0.0, 0.0, 1.0]])

# Aplica a homografia aos pontos
pontos_homog = cv2.perspectiveTransform(pontos_cnn.reshape(-1, 1, 2), H)
print("Pontos transformados:", pontos_homog.reshape(-1, 2))
