import cv2
import numpy as np
import os

#import numba


def create_output_folder(input_folder):
    output_folder = os.path.join(input_folder, "GrayScale")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

def RGB_to_GrayScale(input_folder):
    output_folder = create_output_folder(input_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            input_image = os.path.join(input_folder, filename)
            
            img = cv2.imread(input_image)
            height,width, _=img.shape
            gray_img1 = np.zeros((height, width), dtype=np.uint8)
            gray_img2 = np.zeros((height, width), dtype=np.uint8)

            for i in range(height):
                for j in range(width):
                    gray_img1[i, j] = (img[i, j, 0] + img[i, j, 1] + img[i, j, 2]) / 3
                    gray_img2[i, j] = (img[i, j, 0] * 0.299 + img[i, j, 1] * 0.587 + img[i, j, 2] * 0.114)

            output_image1 = os.path.join(output_folder, os.path.splitext(filename)[0] + "_gray1.jpg")
            output_image2 = os.path.join(output_folder, os.path.splitext(filename)[0] + "_gray2.jpg")
            cv2.imwrite(output_image1, gray_img1)
            cv2.imwrite(output_image2, gray_img2)

input_folder = '/home/mikolaj/GitHub/Detekcja-wizyjna-odpad-w-komunalnych/dataset/butelki/train_data'

RGB_to_GrayScale(input_folder)
print("done")    
    
