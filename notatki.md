Augmentor do obróbki datasetu
SKlearn do uczenia
Pillow/opencv - przygotować zdjęcia jako dane do modeli treningowych

Przystosowanie danych:
Zdjęcia powinny być tego samego rozmiaru, kolorowe
Normalizacja pikseli 0-1. Możesz użyć narzędzi takich jak MinMaxScaler z sklearn.preprocessing w Pythonie.
Obracanie, odbicia lustrzane itp. Biblioteki takie jak ImageDataGenerator w przypadku TensorFlow lub Keras mogą ułatwić ten proces.
Upewnij się, że masz odpowiednie etykiety (znaczniki) dla każdego obrazu, wskazujące, do której klasy należy. Znaczniki te są niezbędne do trenowania modelu nadzorowanego.
Zastosowanie Jąder (w przypadku SVM):
Jeśli używasz Support Vector Machine (SVM) i masz dane, które nie są liniowo separowalne, możesz rozważyć zastosowanie jąder do przekształcenia danych do przestrzeni o wyższej wymiarowości.
Podział na Zbiór Treningowy i Testowy

Działanie na modelu:
	1. import bibliotek:
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
	2. podział na zbiór uczący i testywy (zakładam, ze zrobione)
x_train, y_train, x_test, y_test, gdzie x to dane wejściowe - zdjęcia, a y to dane wyjściowe - etykiety
	3. inicjalizacja i trenowanie modelu
Wybierz model SVM, ustaw parametry, takie jak rodzaj jądra (np. 'linear' dla jądra liniowego), a następnie użyj funkcji fit do trenowania modelu.
model = SVC(kernel='linear', C=1)  # Przykład dla jądra liniowego
model.fit(X_train, y_train)
	4 ocena modelu:
Przewiduj etykiety dla zbioru testowego i porównaj je z rzeczywistymi etykietami, aby ocenić skuteczność modelu.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu: {accuracy}')
