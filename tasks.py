from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import DT_to_PNG

# import des donnees
breast_cancer = load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target


# separation des donnees
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

## task 1 ##
# generation de l'arbre de decision
clf = DecisionTreeClassifier(max_leaf_nodes=2)
clf.fit(x_train, y_train)

# affichage de l'arbre de decision
DT_to_PNG(clf, breast_cancer.feature_names, "./tree task 1")