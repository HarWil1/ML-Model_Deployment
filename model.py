# Importing the necessary packages and model metrics
import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.model_selection import train_test_split # data split
from sklearn.tree import DecisionTreeClassifier # Decision tree algorithm
from sklearn.neighbors import KNeighborsClassifier # KNN algorithm
from sklearn.linear_model import LogisticRegression # Logistic regression algorithm
from sklearn.ensemble import RandomForestClassifier # Random forest tree algorithm
from sklearn.metrics import accuracy_score # evaluation metric
from sklearn.metrics import recall_score # recall metric
from sklearn.model_selection import train_test_split
import pickle
#================================================================

# Load the csv file

df = pd.read_csv(r'C:\Users\TALEHOUSE\Downloads\iris.csv')
df.head()


df.columns

#Index(['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species'],
      

# Rename the cab data column Date of Travel to Date
df = df.rename(columns = {'SepalLengthCm':'Sepal_Length','SepalWidthCm':'Sepal_Width',
                          'PetalLengthCm':'Petal_Length','PetalWidthCm':'Petal_Width',
                          'Species':'Class'})


# Select independent and dependent variable
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df["Class"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)


#======================================
# MODELING STAGE

# Instantiate the models
# 1. RandomForest

classifier1 = RandomForestClassifier()
classifier1.fit(X_train, y_train)

preds = classifier1.predict(X_test)

# Evaluate accuracy
print(accuracy_score(y_test, preds))

#=========================================  
# 2. Decision Tree

classifier2 = DecisionTreeClassifier()
classifier2.fit(X_train, y_train)

preds = classifier2.predict(X_test)

# Evaluate accuracy
print(accuracy_score(y_test, preds))

#=========================================
# 3. Logistic Regression

classifier3= LogisticRegression()
classifier3.fit(X_train, y_train)

preds = classifier3.predict(X_test)

# Evaluate accuracy
print(accuracy_score(y_test, preds))

#=========================================
# 4. K-Nearest Neighbors

#Train/Test K-NN Classifier.
n = 5
classifier4= KNeighborsClassifier()
classifier4.fit(X_train, y_train)

preds = classifier4.predict(X_test)

# Evaluate accuracy
print(accuracy_score(y_test, preds))

#=========================================
# so the best model is Classifier3 which is the LogisticRegression
# Make pickle file of our model
pickle.dump(classifier3, open("model.pkl", "wb"))