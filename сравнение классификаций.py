import zipfile
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor





df = ''
with zipfile.ZipFile(r"C:\Users\диджей арбуз\Downloads\archive (mushroom).zip") as file:   
    df = pd.read_csv(r'C:\Users\диджей арбуз\Desktop\f\mushrooms\mushrooms.csv')
   



X = df[['habitat', 'odor']]
y = df['class']
X = pd.get_dummies(X)
#y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size=0.2)  

lr = LogisticRegression()
lr.fit(x_train, y_train)
pred = lr.predict(x_test)
acc = accuracy_score(y_test, pred)
ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap='Blues')
plt.show() 


class Knn:
    def __init__(self, n_neighbors = 5, regression = False):
        self.n_neighbors = n_neighbors
        self.regression = regression

    def fit(self, x_train, y_train):
        self.x_train, self.y_train =  x_train, y_train


