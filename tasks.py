from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import *

# import des donnees
breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target

# separation des donnees
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

## task 1 ##
# generation de l'arbre de decision
clf1 = DecisionTreeClassifier(max_leaf_nodes=2)
clf1.fit(x_train, y_train)

# affichage de l'arbre de decision (export au format pdf pour le rapport)
DT_to_PNG(clf1, breast_cancer.feature_names, "./tree task 1")
DT_to_PDF(clf1, breast_cancer.feature_names, "./tree task 1")

## task 2 ##
# generation de l'arbre de decision
clf2 = DecisionTreeClassifier(max_leaf_nodes=30)
clf2.fit(x_train, y_train)

# affichage de l'arbre de decision (export au format pdf pour le rapport)
DT_to_PNG(clf2, breast_cancer.feature_names, "./tree task 2")
DT_to_PDF(clf2, breast_cancer.feature_names, "./tree task 2")
