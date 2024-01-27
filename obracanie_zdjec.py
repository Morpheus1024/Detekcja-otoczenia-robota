import os
import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def odczytaj_serie_obrazow(sciezka_do_folderu):
    lista_obrazow = []

    for plik in os.listdir(sciezka_do_folderu):
        if plik.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            sciezka_do_obrazu = os.path.join(sciezka_do_folderu, plik)
            lista_obrazow.append(sciezka_do_obrazu)

    return lista_obrazow


def zmien_rozmiar_obrazow(serie_obrazow, folder_wyjsciowy, nowy_rozmiar=(1080, 720)):
    if not os.path.exists(folder_wyjsciowy):
        os.makedirs(folder_wyjsciowy)

    for idx, sciezka_obrazu in enumerate(serie_obrazow):
        with Image.open(sciezka_obrazu) as obraz:
            # Odczytaj aktualny rozmiar obrazu
            aktualny_rozmiar = obraz.size

            # Dostosuj proporcje, zachowując aspekt obrazu
            nowy_rozmiar = (nowy_rozmiar[0], int((nowy_rozmiar[0] / aktualny_rozmiar[0]) * aktualny_rozmiar[1]))

            # Użyj PyTorch do zmiany rozmiaru obrazu na CPU
            transformer = transforms.Resize(nowy_rozmiar)
            obraz_resized = transformer(obraz)

            # Zapisz zmieniony obraz w formacie JPEG
            nazwa_pliku = f"obraz_{idx + 1}.jpg"
            sciezka_do_zapisu = os.path.join(folder_wyjsciowy, nazwa_pliku)
            obraz_resized.save(sciezka_do_zapisu, format="JPEG")


def obracaj_przeskalowane_obrazy(serie_obrazow, folder_wyjsciowy_obroty, nowy_rozmiar=(1080, 720), kat_obrotu=90):
    if not os.path.exists(folder_wyjsciowy_obroty):
        os.makedirs(folder_wyjsciowy_obroty)

    for idx, sciezka_obrazu in enumerate(serie_obrazow):
        if sciezka_obrazu.lower().endswith('.jpg'):
            nazwa_pliku = f"obraz_{idx + 1}_obrocony_{kat_obrotu}_stopni.jpg"
            sciezka_do_przeskalowanego_obrazu = os.path.join(folder_wyjsciowy_obroty, nazwa_pliku)

            with Image.open(sciezka_obrazu) as obraz:
                # Use PyTorch to resize the image on CPU
                transformer = transforms.Resize(nowy_rozmiar)
                obraz_przeskalowany = transformer(obraz)

                # Use PyTorch to rotate the image on CPU
                transformer_rotate = transforms.functional.rotate
                obraz_obrocony = transformer_rotate(obraz_przeskalowany, kat_obrotu)

                # Save the rotated image to the "obrocone" folder
                sciezka_do_zapisu = os.path.join(folder_wyjsciowy_obroty, nazwa_pliku)
                obraz_obrocony.save(sciezka_do_zapisu, format="JPEG")


if __name__ == "__main__":
    nazwa_folderu = r"C:\Users\Mateusz\PycharmProjects\Detekcja-otoczenia-robota\dataset\butelki\train_data"
    sciezka_do_folderu = os.path.join(os.getcwd(), nazwa_folderu)

    serie_obrazow = odczytaj_serie_obrazow(sciezka_do_folderu)

    if serie_obrazow:
        folder_wyjsciowy_rozmiar = "zmienione_rozmiary"
        zmien_rozmiar_obrazow(serie_obrazow, folder_wyjsciowy_rozmiar, nowy_rozmiar=(1080, 720))
        print(f"Odczytane obrazy zostały zapisane w folderze '{folder_wyjsciowy_rozmiar}' w formacie JPEG.")

        folder_wyjsciowy_obroty = "obrocone_obrazy"
        obracaj_przeskalowane_obrazy(serie_obrazow, folder_wyjsciowy_obroty, kat_obrotu=90)
        obracaj_przeskalowane_obrazy(serie_obrazow, folder_wyjsciowy_obroty, kat_obrotu=-90)
        print(f"Obrocone obrazy zostały zapisane w folderze '{folder_wyjsciowy_obroty}' w formacie JPEG.")
    else:
        print("Brak obrazów do odczytu.")
