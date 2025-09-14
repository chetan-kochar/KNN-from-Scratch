import math as mt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def labelEncoding(x) :
    if x == 'M':
        return 0
    elif x == 'B':
        return 1


def train_test_split(X, Y, train_percent=0.7, test_percent=0.3, shuffle=True, random_state=None):
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        idx = np.random.permutation(len(X))
        X = X.iloc[idx].reset_index(drop=True)
        Y = Y.iloc[idx].reset_index(drop=True)
    lengthX = len(X)
    train_size = int(lengthX * train_percent)
    x_train = X[:train_size]
    y_train = Y[:train_size]
    x_test = X[train_size:]
    y_test = Y[train_size:]
    return x_train, y_train, x_test, y_test

def Normalizer(data):
    # Data must be a numpy array
    max = np.max(data)
    NormalizedData = data / max
    return NormalizedData


class KNNClassifier():
    def __init__(self ,  k=5):
        if(k%2==0):
            k+=1
        self.bestK = k
        self.k = k
    
    def train(self ,x_train , y_train , cols):
        self.x_train = x_train
        self.y_train = y_train
        self.columns = cols

    def predict(self,x_test):
        predictions = []
        for i in range(len(x_test)):   # iterate over test rows
            test_point = np.array([x_test[clm].iloc[i] for clm in self.columns])
            distances = self.euclidean_dist(test_point)
            prediction = self.findClosestNeighbours(distances)
            predictions.append(prediction)  
        return pd.Series(predictions)

    def euclidean_dist(self, test_point):
        Edist = []
        for i in range(len(self.x_train)):
            train_point = np.array([self.x_train[clm].iloc[i] for clm in self.columns])
            dist = mt.sqrt(np.sum((test_point - train_point) ** 2))
            Edist.append(dist)
        return pd.Series(Edist)
        
    def findClosestNeighbours(self,distList):
        k = min(self.k, len(distList))  # Ensure k does not exceed available samples
        distList = distList.sort_values().head(k)
        if distList.empty:
            return None 
        predList = self.y_train.iloc[distList.index].tolist()
        predS = pd.Series(predList)
        if predS.empty:
            return None  
        return predS.value_counts().index[0]

    def accuracy(self,y_true , y_pred):
        Score = 0
        for i in range(len(y_true)):
            if y_true.iloc[i] == y_pred.iloc[i]:
                Score+=1
        return round((Score*100)/len(y_pred),2)

    def best_k(self , x_test , y_test):
        Scores = []
        k_values = []
        max_k = min(len(self.x_train), 15)  # Don't test k larger than training set
        for i in range(2, min(9, (max_k//2)+1)):
            k = 2 * i - 1
            self.k = k
            y_pred = self.predict(x_test)
            acc = self.accuracy(y_test, y_pred)
            Scores.append(acc)
            k_values.append(k)
            print(f"Tested k={k}, Accuracy={acc}")
        self.bestK = k_values[Scores.index(max(Scores))]
        plt.plot(k_values , Scores)
        plt.xlabel("K values")
        plt.ylabel("Scores")
        plt.title("Custom KNN")
        plt.show()
        return self.bestK


dataset = pd.read_csv("Datasets\\KNN\\breast-cancer.csv").drop("id",axis=1)
cols = ['radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']

X = dataset.iloc[:,1:11] #Selected only the first 2 features for training
Y = dataset["diagnosis"].map(labelEncoding) # Output -> 0 : 'B',  1 : 'M'

x_train, y_train, x_test, y_test = train_test_split(X,Y)

model = KNNClassifier()
model.train(x_train ,y_train , cols)
bestK = model.best_k(x_test,y_test)
print("Best k : " , bestK)

best_model = KNNClassifier(k = bestK)
best_model.train(x_train,y_train ,cols)
y_pred1 = best_model.predict(x_test)
accuracy1 = best_model.accuracy(y_test , y_pred1)
print("Accuracy is ",accuracy1)
print("#"*100)



##################################################################################################################################################
#Using SKLEARN Library
import math as mt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix , ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

def labelEncoding(x) :
    if x == 'M':
        return 0
    elif x == 'B':
        return 1

# Loaded dataset properly
dataset = pd.read_csv("Datasets\\KNN\\breast-cancer.csv").drop("id",axis=1)
X = dataset.iloc[:,1:11] #Selected only the first 2 features for training
Y = dataset["diagnosis"].map(labelEncoding) # Output -> 0 : 'B',  1 : 'M'

# Train test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# Creating a model instance
knn_model = KNeighborsClassifier() # Default k = 5

# Finding best value of k
score = []
k_values = []
for i in range(2,9):
    k = 2*i -1 #Because value of k should always be odd
    k_values.append(k)
    test_knn_model = KNeighborsClassifier(k)
    # Training the model
    test_knn_model.fit(x_train, y_train)
    # Getting the accuracy score
    score.append(round(test_knn_model.score(x_test,y_test),2))
print("SKLEARN Scores",score)
print("SKLEARN k" , k_values)
# Plotting to visaulize performance of model with varied value of k
plt.plot(k_values,score)
plt.xlabel("k")
plt.ylabel("Score")
plt.title("Finding best k value using SKLEARN")
plt.show()


# Creating a model with best k
k = 5
knn_model = KNeighborsClassifier(n_neighbors = k)
knn_model.fit(x_train,y_train)
y_pred = knn_model.predict(x_test)
score = knn_model.score(x_test , y_test) # KNN 's own accuracy score function
accuracy_score = accuracy_score(y_test,y_pred) # SKLearn's acccuracy function
print("KNN Score",round(score*100,2))
print("SKlearn Accuracy Score",round(accuracy_score*100,2))

# Calculating Confusion Matrix
CM = confusion_matrix(y_test , y_pred) #2x2 CM will be created
# Visualizing the CM
CM_Dis = ConfusionMatrixDisplay(CM)
CM_Dis.plot()
plt.show()

# Plotting training data and test predictions
plt.figure(figsize=(8,6))
# Plot training data (dots)
plt.scatter(x_train.iloc[:,0], x_train.iloc[:,1], c=y_train, cmap='coolwarm', marker='o', label='Train')
# Plot test data predictions (stars)
plt.scatter(x_test.iloc[:,0], x_test.iloc[:,1], c=y_pred, cmap='coolwarm', marker='*', s=150, edgecolor='k', label='Test Prediction')
plt.xlabel(x_train.columns[0])
plt.ylabel(x_train.columns[1])
plt.title('KNN Classification: Train (dots) vs Test Prediction (stars)')
plt.legend()
plt.show()