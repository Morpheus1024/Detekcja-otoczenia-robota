import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Ustawienie dostępności GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Wczytaj wytrenowany model ResNet50
model = ResNet50(weights='imagenet')

def predict_top_classes(img, top_k=5):
    # Zmień rozmiar obrazu na 224x224 pikseli
    img = cv2.resize(img, (224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Dokonaj predykcji za pomocą modelu
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=top_k)

    # Wyświetl top k przewidywanych klas
    top_classes = []
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
        print(f"{i + 1}: {label} ({score:.2f})")
        top_classes.append(label)

    return top_classes

def find_bottle_contour(img):
    # Konwersja obrazu do skali szarości
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detekcja krawędzi za pomocą algorytmu Canny'ego
    edges = cv2.Canny(gray, 50, 500)

    # Znajdź kontury w obrazie krawędzi
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Wybierz największy kontur (zakładając, że to jest butelka)
    if contours:
        bottle_contour = max(contours, key=cv2.contourArea)
        return bottle_contour
    else:
        return None

def draw_contour_bounding_box(img, contour):
    if contour is not None:
        # Znajdź prostokątny bounding box dla konturu butelki
        x, y, w, h = cv2.boundingRect(contour)

        # Narysuj prostokątny bounding box na obrazie
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img

# Ścieżka do folderu z obrazami
folder_path = 'obrocone_obrazy'
output_folder_path = 'bounding_boxy'

# Utwórz folder "bounding_boxy", jeśli nie istnieje
os.makedirs(output_folder_path, exist_ok=True)

# Przetwarzaj każdy plik w folderze
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Akceptuj tylko pliki obrazowe
        image_path = os.path.join(folder_path, filename)

        # Wczytaj obraz
        img = cv2.imread(image_path)

        # Przewiduj top 5 klas obrazu
        top_classes = predict_top_classes(img)

        # Znajdź kontur butelki
        bottle_contour = find_bottle_contour(img)

        # Rysuj bounding box na podstawie konturu i zapisz zmodyfikowany obraz w folderze "bounding_boxy"
        img_with_box = draw_contour_bounding_box(img.copy(), bottle_contour)
        output_path = os.path.join(output_folder_path, f"annotated_{filename}")
        cv2.imwrite(output_path, img_with_box)

        print(f"Zapisano obraz z bounding boxem: {output_path}")
