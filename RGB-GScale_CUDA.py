import cv2
import numpy as np
import os
from numba import cuda

def create_output_folder(input_folder):
    output_folder = os.path.join(input_folder, "GrayScaleCUDA")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

@cuda.jit
def rgb_to_grayscale_cuda(input_image, output_image):
    i, j = cuda.grid(2)
    height, width, _ = input_image.shape
    
    if i < height and j < width:
        # Oblicz wartość odcienia szarości
        gray_value = 0.299 * input_image[i, j, 0] + 0.587 * input_image[i, j, 1] + 0.114 * input_image[i, j, 2]
        # Ustaw wartość w obrazie wynikowym
        output_image[i, j] = gray_value

@cuda.jit
def resize_cuda(input_image, output_image, new_height, new_width):
    i, j = cuda.grid(2)
    height, width, _ = input_image.shape
    
    if i < new_height and j < new_width:
        # Mapowanie współrzędnych pikseli na oryginalny rozmiar
        orig_i = i * height # new_height
        orig_j = j * width # new_width
        # Przypisanie wartości do nowego obrazu
        output_image[i, j, 0] = input_image[orig_i, orig_j, 0]
        output_image[i, j, 1] = input_image[orig_i, orig_j, 1]
        output_image[i, j, 2] = input_image[orig_i, orig_j, 2]

def RGB_GSCALE_CUDA(filename, output):
    img = cv2.imread(filename)
    height, width, _ = img.shape
    
    input_image_gpu = cuda.to_device(img)
    
    new_height, new_width = 720, 1280
    output_image_gpu = cuda.device_array((new_height, new_width, 3), dtype=np.uint8)
    
    threads_per_block = (16, 16)
    blocks_per_grid_x = (new_height + threads_per_block[0] - 1) # threads_per_block[0]
    blocks_per_grid_y = (new_width + threads_per_block[1] - 1) # threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    rgb_to_grayscale_cuda[blocks_per_grid, threads_per_block](input_image_gpu, output_image_gpu)
    
    resized_image_gpu = cuda.device_array((new_height, new_width, 3), dtype=np.uint8)
    
    resize_cuda[blocks_per_grid, threads_per_block](output_image_gpu, resized_image_gpu, new_height, new_width)
    
    resized_img_result = resized_image_gpu.copy_to_host()
    
    output_image_result = os.path.join(output, os.path.splitext(os.path.basename(filename))[0] + "_gray_cuda_resized.jpg")
    cv2.imwrite(output_image_result, resized_img_result)


input_folder = '/home/mikolaj/GitHub/Detekcja-wizyjna-odpad-w-komunalnych/dataset/butelki/train_data'
output = create_output_folder(input_folder)

for filename in os.listdir(input_folder):
    filename = os.path.join(input_folder, filename)
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        RGB_GSCALE_CUDA(filename, output)

print("done")
