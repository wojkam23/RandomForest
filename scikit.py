from main import RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3
)

# Model scikit-learn
clf = RandomForestClassifier(n_estimators=5, max_depth=2)
clf.fit(X_train, y_train)
sklearn_predictions = clf.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

# napisany model
my_model = RandomForest(n_estimators=5, max_depth=2)
my_model.fit(X_train, y_train)
my_predictions = my_model.predict(X_test)
my_accuracy = accuracy_score(y_test, my_predictions)

print(f"Dokładność modelu scikit-learn: {sklearn_accuracy}")
print(f"Dokładność napisanego modelu: {my_accuracy}")
