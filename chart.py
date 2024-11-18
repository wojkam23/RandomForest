import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from main import RandomForest

# Załadowanie zbioru danych Iris
iris = load_iris()
X = iris.data[:, :2]  # Używamy dwóch pierwszych cech (sepal length, sepal width)
y = iris.target

# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening napisanego modelu
my_model = RandomForest(n_estimators=10, max_depth=2)
my_model.fit(X_train, y_train)
my_predictions = my_model.predict(X)

# Trening modelu scikit-learn
sklearn_model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
sklearn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_model.predict(X)

# siatka punktów do wizualizacji
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Przewidywania dla siatki punktów (napisany model)
Z_my_model = my_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_my_model = np.array(Z_my_model).reshape(xx.shape)

# Przewidywania dla siatki punktów (scikit-learn)
Z_sklearn_model = sklearn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_sklearn_model = Z_sklearn_model.reshape(xx.shape)


plt.figure(figsize=(12, 6))

# Napisany model
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_my_model, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Set1)
plt.title("Wyniki - stworzony model")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

# Model scikit-learn
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_sklearn_model, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Set1)
plt.title("Wyniki - scikit-learn")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

# Wyświetlanie wykresu
plt.tight_layout()
plt.show()
