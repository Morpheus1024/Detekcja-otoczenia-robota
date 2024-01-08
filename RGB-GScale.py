import cv2
import numpy as np
import os

# CPU: Ryzen 7 5800H
# czas działania dla datasetu butelek (63 pliki) to ok. 45 minut

def create_output_folder(input_folder):
    output_folder = os.path.join(input_folder, "GrayScale")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

def RGB_GSCALE(filename, output):
    img = cv2.imread(filename)
    height, width, _ = img.shape
    gray_img1 = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            gray_img1[i, j] = (img[i, j, 0] * 0.299 + img[i, j, 1] * 0.587 + img[i, j, 2] * 0.114)

    resized_img1 = cv2.resize(gray_img1, (720,1280))

    output_image1 = os.path.join(output, os.path.splitext(filename)[0] + "_gray1.jpg")

    cv2.imwrite(output_image1, resized_img1)

input_folder = '/home/mikolaj/GitHub/Detekcja-wizyjna-odpad-w-komunalnych/dataset/butelki/train_data'
output = create_output_folder(input_folder)

for filename in os.listdir(input_folder):
    filename = os.path.join(input_folder, filename)
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        RGB_GSCALE(filename, output)

print("done")


    
