from matplotlib import colors
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import *
import matplotlib.pyplot as plt

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

# score de l'arbre de decision
clf1_training_score = clf1.score(x_train, y_train)
clf1_testing_score = clf1.score(x_test, y_test)

# affichage de l'arbre de decision (export au format pdf pour le rapport)
DT_to_PDF(clf1, breast_cancer.feature_names, "./tree task 1")

## task 2 ##
# generation de l'arbre de decision
clf2 = DecisionTreeClassifier(max_leaf_nodes=30)
clf2.fit(x_train, y_train)

# score de l'arbre de decision
clf2_training_score = clf2.score(x_train, y_train)
clf2_testing_score = clf2.score(x_test, y_test)

# affichage de l'arbre de decision (export au format pdf pour le rapport)
DT_to_PDF(clf2, breast_cancer.feature_names, "./tree task 2")

## task 3 ##
# ensembles des arbres de decision
trees = [None]*29
trees[0] = clf1
for i in range(3,31):
    
    # generation de l'arbre de decision de profondeur i
    trees[i-2] = DecisionTreeClassifier(max_leaf_nodes=i)
    trees[i-2].fit(x_train, y_train)

# training et testing scores
training_scores = []
testing_scores = []
for tree in trees:
    training_scores += [tree.score(x_train, y_train)]
    testing_scores += [tree.score(x_test, y_test)]

# affichage des resultats
## 1. evolution des scores
max_leaf_nodes = range(2,31)
cm = 1/2.54
fig = plt.figure(figsize=(16*cm, 10*cm))
plt.plot(max_leaf_nodes, training_scores, label="training scores")
plt.hlines(y = max(testing_scores), xmin = max_leaf_nodes[0], xmax = max_leaf_nodes[-1], 
           colors='r', linestyles='dashed', label="maximum testing score")
plt.plot(max_leaf_nodes, testing_scores, label="testing score")
plt.ylabel("Score")
plt.xlabel("Maximal number of nodes")
plt.legend(loc='upper left')
plt.savefig("training and testing score.pdf", bbox_inches='tight')

## 2. profondeur de l'arbre en fonction du nombre maximal de noeuds
depths = []
for tree in trees:
    depths += [tree.get_depth()]
y_max = max(depths)    
x_max = max_leaf_nodes[depths.index(y_max)]
fig = plt.figure(figsize=(16*cm, 7*cm))
plt.plot(max_leaf_nodes, depths)
plt.scatter(x_max, y_max, s=80, facecolors='none', edgecolors='r')
plt.annotate('16 nodes', xy=(x_max+x_max/40,y_max-y_max/15), color='r')
plt.ylabel("Depth")
plt.xlabel("Maximal number of nodes")
plt.savefig("trees depth.pdf", bbox_inches='tight')