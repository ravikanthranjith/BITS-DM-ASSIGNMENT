# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

col_names = ['GeneID','Colour','Cell_Cycle_Cat','Cell_Cycle_Number','Gene_Induction','Symbol','LocusTag','Synonyms','dbXrefs','chromosome','description','type_of_gene','Other_designations','Class']

# load dataset
pima = pd.read_csv("SampleData1.csv", header=None, names=col_names)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in range(13):
    pima[:,i] = le.fit_transform(pima[:,i])

pima.head()

#split dataset in features and target variable
feature_cols = ['Colour','Cell_Cycle_Cat','Cell_Cycle_Number','Gene_Induction','Symbol','LocusTag','Synonyms','dbXrefs','chromosome','description','type_of_gene','Other_designations']

X = pima[feature_cols] # Features
y = pima.Class # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
