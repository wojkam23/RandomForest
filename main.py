import numpy as np
from collections import Counter

# Klasa, która implementuje drzewo decyzyjne
class DecisionTree:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.tree = None
        self.feature_splits = []

    # Budowa drzewa decyzyjnego
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    # Przewidywanie na podstawie drzewa
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    # Warunek zakończenia budowy drzewa
    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(set(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        # Losowy wybór cechy i podziału
        feature = np.random.randint(0, X.shape[1])
        threshold = float(np.median(X[:, feature]))

        self.feature_splits.append((feature, threshold))

        # Dzielimy dane na lewą i prawą gałąź
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return Counter(y).most_common(1)[0][0]

        # Rekurencyjna budowa drzewa
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return (feature, threshold, left_subtree, right_subtree)

    # Przechodzenie po drzewie w celu przewidywania
    def _traverse_tree(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left_subtree, right_subtree = node
        if x[feature] <= threshold:
            return self._traverse_tree(x, left_subtree)
        else:
            return self._traverse_tree(x, right_subtree)

    # Zwraca informacje o podziałach cech i liczbie podziałów
    def get_tree_info(self):
        return {
            "depth": self.max_depth,
            "splits": len(self.feature_splits),
            "features_used": self.feature_splits
        }


# Klasa, która implementuje las losowy
class RandomForest:
    def __init__(self, n_estimators=5, max_depth=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    # Trenowanie lasu losowego na danych X i y
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    # głosowanie większościowe
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]

    # tworzenie próbki bootstrapowej z danych
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples // 2, replace=True)
        return X[indices], y[indices]

    # Funkcja, która wyświetla informacje o wszystkich drzewach w lesie
    def display_forest_info(self):
        print("Informacje o drzewach w lesie:")
        for i, tree in enumerate(self.trees):
            tree_info = tree.get_tree_info()
            print(f"Drzewo {i + 1}:")
            print(f" - Głębokość: {tree_info['depth']}")
            print(f" - Liczba podziałów: {tree_info['splits']}")
            print(f" - Użyte cechy i progi podziałów: {tree_info['features_used']}")
            print("-" * 40)