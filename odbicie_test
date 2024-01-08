!pip install numba
!pip install scikit-cuda

#wymaga '/content/input/your_image.jpg'

import numpy as np
import cv2
import numba
from numba import cuda
import matplotlib.pyplot as plt

# Załaduj obraz z folderu 'input'
image_path = '/content/input/1704391748750.jpg'
image = cv2.imread(image_path)

# Funkcja do odbicia lustrzanego obrazu za pomocą CUDA
@cuda.jit
def flip_image_gpu(input_img, output_img, horizontal):
    x, y = cuda.grid(2)
    rows, cols = input_img.shape[:2]
    
    # Oblicz nowe współrzędne piksela po odbiciu lustrzanym
    new_x = cols - x - 1 if horizontal else x
    new_y = rows - y - 1 if not horizontal else y
    
    # Sprawdź, czy nowe współrzędne są w granicach obrazu
    if 0 <= new_x < cols and 0 <= new_y < rows:
        output_img[y, x, 0] = input_img[new_y, new_x, 0]
        output_img[y, x, 1] = input_img[new_y, new_x, 1]
        output_img[y, x, 2] = input_img[new_y, new_x, 2]

# Przygotuj obraz do odbicia lustrzanego
horizontal_flip = True  # Odbicie lustrzane w poziomie (możesz zmienić na False dla odbicia pionowego)
output_img = np.zeros_like(image)

# Konfiguracja rozmiaru bloku i siatki CUDA
threadsperblock = (16, 16)
blockspergrid_x = int(np.ceil(image.shape[1] / threadsperblock[0]))
blockspergrid_y = int(np.ceil(image.shape[0] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)

# Skopiuj obraz na GPU
d_image = cuda.to_device(image)
d_output_img = cuda.to_device(output_img)

# Wywołaj funkcję CUDA do odbicia lustrzanego obrazu
flip_image_gpu[blockspergrid, threadsperblock](d_image, d_output_img, horizontal_flip)

# Skopiuj wynik z powrotem na CPU
d_output_img.copy_to_host(output_img)

# Wyświetl oryginalny i obrócony obraz
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
plt.title('Mirrored Image')

plt.show()
