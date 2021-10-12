from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import DT_to_PNG

# import des donnees
data = load_breast_cancer(as_frame=True)

# separation des donnees
training_data, testing_data = train_test_split(data.data, test_size=0.33, random_state=42)
print(training_data)
print(testing_data)