import cv2
import numpy as np
import os
## czas wykonania na CPU to około 1 minuta

output_path = '/home/mikolaj/GitHub/Detekcja-wizyjna-odpad-w-komunalnych/dodanie_szumow'  # Specify the output folder
dataset_path = '/home/mikolaj/GitHub/Detekcja-wizyjna-odpad-w-komunalnych/dataset/butelki/train_data'

def add_noise(image, mean=0, sigma=100):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = cv2.resize(noisy, (1280, 720))
    return np.clip(noisy, 0, 255).astype(np.uint8)

for filename in os.listdir(dataset_path):
    filename = os.path.join(dataset_path, filename)
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        image = cv2.imread(filename)
        noise_image = add_noise(image)
        output_photo = os.path.join(output_path, os.path.splitext(os.path.basename(filename))[0] + "_noise.jpg")
        cv2.imwrite(output_photo, noise_image) 
