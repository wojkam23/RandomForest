# Random Forest  Python

## Opis projektu

Projekt ten zawiera implementację algorytmu **Random Forest** języku Python oraz porównanie wyników z biblioteką **scikit-learn**. Implementacja obejmuje:

- **Drzewo decyzyjne (DecisionTree)**: Implementacja drzewa decyzyjnego z losowym wyborem cechy i użyciem mediany jako progu podziału.
- **Las losowy (RandomForest)**: Klasa, która tworzy wiele drzew decyzyjnych i łączy wyniki za pomocą głosowania większościowego.
- **Porównanie**: W projekcie porównujemy działanie własnej implementacji z modelem Random Forest z biblioteki scikit-learn na zbiorze danych Iris.

## Zawartość projektu

- `main.py`: Implementacja drzewa decyzyjnego i lasu losowego od podstaw.
- `test_random_forest.py`: Testy jednostkowe dla implementacji lasu losowego.
- `iris.py`: Przykład użycia napisanego modelu Random Forest na zbiorze danych Iris.
- `scikit.py`: Porównanie wyników napisanego modelu z modelem z biblioteki scikit-learn.
- `chart.py`: Wizualizacja granic decyzji dla napisanego modelu i modelu z scikit-learn na dwóch cechach zbioru danych Iris.

  ├── main.py                
├── test_random_forest.py  
├── iris.py                
├── scikit.py              
├── chart.py               
├── README.md              


## Zbiór danych

Użyto zbioru danych **Iris**, który zawiera informacje o trzech gatunkach irysów:
- `sepal length` (długość działki kielicha)
- `sepal width` (szerokość działki kielicha)
- `petal length` (długość płatka)
- `petal width` (szerokość płatka)

Zbiór danych jest dostępny w bibliotece **scikit-learn** i składa się z 150 próbek.

## Wyniki działania

### 1. Porównanie dokładności

Poniżej znajduje się porównanie dokładności napisanego modelu oraz modelu z biblioteki **scikit-learn**:


![Porównanie dokładności](images/accuracy_comparison.png)
<img width="398" alt="accuracy_comparison" src="https://github.com/user-attachments/assets/52958682-70e6-4854-a378-040eae14e0ae">


Jak widać, napisany model osiągnął dokładność **0.8**, podczas gdy model scikit-learn osiągnął dokładność **0.93**. Różnica ta wynika z implementacji stworzonego modelu, który stosuje losowe podziały i ma mniejszą liczbę estymatorów.

### 2. Informacje o drzewach w lesie

Własna implementacja lasu losowego zawiera szczegóły dotyczące każdego drzewa. Poniżej znajduje się przykładowy wynik wyświetlający informacje o drzewach:

![Informacje o drzewach](images/tree_info.png)

Dla każdego drzewa wyświetlane są następujące informacje:
- **Głębokość**: Maksymalna głębokość drzewa (ustawiona na 2).
- **Liczba podziałów**: Liczba wykonanych podziałów w drzewie.
- **Użyte cechy i progi podziałów**: Lista cech i progów, które zostały użyte do podziału danych.

### 3. Wizualizacja granic decyzji

Na poniższym wykresie przedstawiono porównanie granic decyzji dla napisanego modelu oraz modelu z biblioteki scikit-learn:

![Granice decyzji](images/decision_boundaries.png)

- **Lewy wykres**: Granice decyzji napisanego modelu Random Forest.
- **Prawy wykres**: Granice decyzji modelu Random Forest z biblioteki scikit-learn.

Oba modele klasyfikują dane na zbiorze Iris, używając dwóch pierwszych cech: długości i szerokości działki kielicha. Widać, że model scikit-learn tworzy bardziej złożone granice, co wynika z bardziej zaawansowanych metod podziału i większej liczby estymatorów.



### 4. Unnittesty
![Unittest](images/unittest.png)

Na grafice można zaobserwować wynik uruchomienia testów jednostkowych. Testy weryfikują poprawność działania modelu, sprawdzając trenowanie modelu bez błędów, prawidłowe przewidywanie wyników oraz osiągnięcie akceptowalnej dokładności (≥ 70%).


## Bibliografia
- Colab Codes: "Implementing Random Forests in Python on Iris Dataset"
https://www.colabcodes.com/post/implementing-random-forests-in-python-on-iris-dataset
- Medium Article: "Random Forest Classifier from Scratch"
https://medium.com/@poleakchanrith/random-forest-classifier-implementation-from-scratch-with-python-8cee705624e7
- Machine Learning Mastery: "Random Forest from Scratch in Python" 
https://machinelearningmastery.com/implement-random-forest-scratch-python/
- Data Camp: "Random Forest Classification with Scikit-Learn"
https://www.datacamp.com/tutorial/random-forests-classifier-python
- Scikit-learn Documentation: "Random Forest Classifier"
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Scikit-learn: "The Iris Dataset"
https://scikit-learn.org/1.5/auto_examples/datasets/plot_iris_dataset.html

