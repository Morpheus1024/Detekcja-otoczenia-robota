import os
from PIL import Image
from math import radians, cos, sin

def odczytaj_serie_obrazow(sciezka_do_folderu):
    lista_obrazow = []

    for plik in os.listdir(sciezka_do_folderu):
        if plik.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            sciezka_do_obrazu = os.path.join(sciezka_do_folderu, plik)
            lista_obrazow.append(sciezka_do_obrazu)

    return lista_obrazow


def zmien_rozmiar_obrazow(serie_obrazow, folder_wyjsciowy, nowy_rozmiar=(720, 1080)):
    if not os.path.exists(folder_wyjsciowy):
        os.makedirs(folder_wyjsciowy)

    for idx, sciezka_obrazu in enumerate(serie_obrazow):
        with Image.open(sciezka_obrazu) as obraz:
            # Ręczna zmiana rozmiaru obrazu bez dodawania białego tła
            obraz_resized = resize_image(obraz, nowy_rozmiar)

            # Zapis obrazu o nowym rozmiarze w formacie JPEG
            nazwa_pliku = f"obraz_{idx + 1}.jpg"
            sciezka_do_zapisu = os.path.join(folder_wyjsciowy, nazwa_pliku)
            obraz_resized.save(sciezka_do_zapisu, format="JPEG")


def resize_image(image, new_size):
    width, height = image.size
    new_width, new_height = new_size

    # Ustalam współczynniki zmiany rozmiaru dla każdego wymiaru
    scale_width = new_width / width
    scale_height = new_height / height

    # Tworzę nowy obraz o docelowych rozmiarach
    resized_image = Image.new("RGB", new_size)

    for y in range(new_height):
        for x in range(new_width):
            # Współrzędne punktu w oryginalnym obrazie
            x_original = int(x / scale_width)
            y_original = int(y / scale_height)

            # Pobieram piksel z oryginalnego obrazu i wstawiam do nowego obrazu
            piksel_pobrany = image.getpixel((x_original, y_original))
            resized_image.putpixel((x, y), piksel_pobrany)

    return resized_image


def obracaj_przeskalowane_obrazy(serie_obrazow, folder_wyjsciowy_obroty, nowy_rozmiar=(720, 1080), kat_obrotu=90):
    if not os.path.exists(folder_wyjsciowy_obroty):
        os.makedirs(folder_wyjsciowy_obroty)

    for idx, sciezka_obrazu in enumerate(serie_obrazow):
        # Sprawdzam, czy obraz jest w formacie JPEG
        if sciezka_obrazu.lower().endswith('.jpg'):
            nazwa_pliku = f"obraz_{idx + 1}_obrocony_{kat_obrotu}_stopni.jpg"
            sciezka_do_przeskalowanego_obrazu = os.path.join(folder_wyjsciowy_obroty, nazwa_pliku)

            with Image.open(sciezka_obrazu) as obraz:
                # Przeskalowanie obrazu
                obraz_przeskalowany = resize_image(obraz, nowy_rozmiar)

                # Obracanie obrazu o zadaną ilość stopni (ręczna implementacja)
                obraz_obrocony = obroc_obraz(obraz_przeskalowany, kat_obrotu)

                # Zapis obroconego obrazu do folderu "obrocone"
                sciezka_do_zapisu = os.path.join(folder_wyjsciowy_obroty, nazwa_pliku)
                obraz_obrocony.save(sciezka_do_zapisu, format="JPEG")


def obroc_obraz(image, kat_obrotu):
    # Konwersja kąta na radiany
    kat_rad = radians(kat_obrotu)

    # Pobranie rozmiarów obrazu
    width, height = image.size

    # Obliczenie nowych rozmiarów obrazu po obrocie
    new_width = int(abs(width * cos(kat_rad)) + abs(height * sin(kat_rad)))
    new_height = int(abs(width * sin(kat_rad)) + abs(height * cos(kat_rad)))

    # Tworzenie nowego obrazu o obliczonych rozmiarach
    obraz_obrocony = Image.new("RGB", (new_width, new_height))

    # Obliczenie środka obrazu
    center_x, center_y = new_width // 2, new_height // 2

    for y in range(new_height):
        for x in range(new_width):
            # Obliczenie współrzędnych piksela w oryginalnym obrazie po obrocie
            x_original = int((x - center_x) * cos(-kat_rad) - (y - center_y) * sin(-kat_rad) + width // 2)
            y_original = int((x - center_x) * sin(-kat_rad) + (y - center_y) * cos(-kat_rad) + height // 2)

            # Sprawdzenie, czy obliczone współrzędne mieszczą się w oryginalnych rozmiarach obrazu
            if 0 <= x_original < width and 0 <= y_original < height:
                # Pobranie piksela z oryginalnego obrazu i wstawienie do obrazu obroconego
                piksel_pobrany = image.getpixel((x_original, y_original))
                obraz_obrocony.putpixel((x, y), piksel_pobrany)

    return obraz_obrocony



if __name__ == "__main__":
    nazwa_folderu = r"C:\Users\Mateusz\PycharmProjects\Detekcja-otoczenia-robota\dataset\puszki\train_data"
    sciezka_do_folderu = os.path.join(os.getcwd(), nazwa_folderu)

    serie_obrazow = odczytaj_serie_obrazow(sciezka_do_folderu)

    if serie_obrazow:
        folder_wyjsciowy_rozmiar = "zmienione_rozmiary"
        zmien_rozmiar_obrazow(serie_obrazow, folder_wyjsciowy_rozmiar, nowy_rozmiar=(720, 1080))
        print(f"Odczytane obrazy zostały zapisane w folderze '{folder_wyjsciowy_rozmiar}' w formacie JPEG.")

        folder_wyjsciowy_obroty = "obrocone_obrazy"
        obracaj_przeskalowane_obrazy(serie_obrazow, folder_wyjsciowy_obroty, kat_obrotu=90)
        obracaj_przeskalowane_obrazy(serie_obrazow, folder_wyjsciowy_obroty, kat_obrotu=-90)
        print(f"Obrocone obrazy zostały zapisane w folderze '{folder_wyjsciowy_obroty}' w formacie JPEG.")
    else:
        print("Brak obrazów do odczytu.")
