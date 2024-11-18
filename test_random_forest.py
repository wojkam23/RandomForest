import unittest
import numpy as np
from main import RandomForest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Klasa, która implementuje testy jednostkowe
class TestRandomForest(unittest.TestCase):
    # Funkcja, która przygotowuje dane i model przed każdym testem
    def setUp(self):
        self.iris = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.iris.data, self.iris.target, test_size=0.3
        )
        self.model = RandomForest(n_estimators=10, max_depth=2)

    # Funkcja, która testuje, czy model można wytrenować bez błędów
    def test_fit(self):
        try:
            self.model.fit(self.X_train, self.y_train)
        except Exception as e:
            self.fail(f"Metoda fit() rzuciła wyjątek: {e}")

    # Funkcja, która testuje, czy metoda predict() zwraca odpowiednią liczbę przewidywań
    def test_predict(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

    # Funkcja, która sprawdza, czy dokładność modelu jest na akceptowalnym poziomie (≥ 70%)
    def test_accuracy(self):
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        accuracy = np.mean(predictions == self.y_test)
        self.assertGreaterEqual(accuracy, 0.7)

if __name__ == '__main__':
    unittest.main()
