# import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Ustawienie dostępności GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

print(physical_devices)

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
        print("No Countur")
        return None

def draw_contour_bounding_box(img, contour):
    if contour is not None:
        # Znajdź prostokątny bounding box dla konturu butelki
        x, y, w, h = cv2.boundingRect(contour)

        # Narysuj prostokątny bounding box na obrazie
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


# wczytanie filmu z servera:

ip_address = '192.168.0.82'
# przechwycenie video
cap = cv2.VideoCapture(f"http://{ip_address}:12344/video_feed")

#Czy udało się otworzyć strumień

if not cap.isOpened():
    print("Failed to open camera")
    exit()

#odczyt
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from camera")
        break

    # operacje na klatce
    top_clases = predict_top_classes(img = frame)

    bottle_contour = find_bottle_contour(img = frame)

    farme_with_b_box = draw_contour_bounding_box(img = frame,contour = bottle_contour)

    cv2.imshow("Camera", farme_with_b_box)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()



