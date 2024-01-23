import cv2
import numpy as np
import os
import numba
from numba import cuda

output_path = '/home/mikolaj/GitHub/Detekcja-wizyjna-odpad-w-komunalnych/dodanie_szumow'  # Specify the output folder
dataset_path = '/home/mikolaj/GitHub/Detekcja-wizyjna-odpad-w-komunalnych/dataset/butelki/train_data'

@cuda.jit
def add_noise_cuda(image, noisy, mean, sigma):
    row, col, ch = image.shape
    tx, ty = cuda.grid(2)
    
    if tx < row and ty < col:
        for c in range(ch):
            noisy[tx, ty, c] = image[tx, ty, c] + sigma * cuda.local_rng() + mean
            noisy[tx, ty, c] = max(0, min(255, noisy[tx, ty, c]))

for filename in os.listdir(dataset_path):
    file_path = os.path.join(dataset_path, filename)
    if file_path.lower().endswith((".jpg", ".png", ".jpeg")):
        image = cv2.imread(file_path)
        if image is None:
            print(f"Unable to read image: {file_path}")
            continue

        row, col, ch = image.shape
        noisy = np.empty_like(image)

        threadsperblock = (16, 16)
        blockspergrid_x = (row + threadsperblock[0] - 1) // threadsperblock[0]
        blockspergrid_y = (col + threadsperblock[1] - 1) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        add_noise_cuda[blockspergrid, threadsperblock](image, noisy, 0, 100)

        output_photo = os.path.join(output_path, os.path.splitext(os.path.basename(file_path))[0] + "_noise.jpg")
        cv2.imwrite(output_photo, noisy)
        print(f"Noisy image saved at {output_photo}")

print("Image processing complete.")
