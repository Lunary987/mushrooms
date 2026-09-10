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
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesClassifier 

df = ''
with zipfile.ZipFile(r"C:\Users\диджей арбуз\Downloads\archive (mushroom).zip") as file:   
    df = pd.read_csv(r'C:\Users\диджей арбуз\Desktop\f\mushrooms\mushrooms.csv')
   



# X = df[['habitat', 'odor', 'ring-number', 'cap-color', 'cap-surface']]
# y = df['class']
# X = pd.get_dummies(X)
# #y = pd.get_dummies(y)

# x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size=0.2)  

# lr = LogisticRegression()
# lr.fit(x_train, y_train)
# pred = lr.predict(x_test)
# acc = accuracy_score(y_test, pred)
# cr = cross_val_score(lr, X, y )
# ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap='Blues')
# #plt.show() 
# print(acc)
# print(cr)

# class Knn:
#     def __init__(self, n_neighbors = 5, regression = False):
#         self.n_neighbors = n_neighbors
#         self.regression = regression

#     def fit(self, x_train, y_train):
#         self.x_train, self.y_train =  x_train, y_train

#     def _euclidean_distances(self, x_test_i):
#         return np.sqrt(np.sum((self.x_train - x_test_i) ** 2, axis=1))

#     def _make_prediction(self, x_test_i):
#         distances = self._euclidean_distances(x_test_i)   # distances to all neighbors
#         k_nearest_indexes = np.argsort(distances)[:self.n_neighbors]
#         targets = self.y_train[k_nearest_indexes]   # k-nearest neighbors target values

#         return np.mean(targets) if self.regression else np.bincount(targets).argmax()

#     def predict(self, X_test):
#         return np.array([self._make_prediction(x) for x in X_test])
     
#     def decision_boundary_plot(X, y, x_train, y_train, clf, feature_indexes, title=None):
#         feature1_name, feature2_name = X.columns[feature_indexes]
#         X_feature_columns = X.values[:, feature_indexes]
#         X_train_feature_columns = x_train[:, feature_indexes]
#         clf.fit(X_train_feature_columns, y_train)

#         plot_decision_regions(X=X_feature_columns, y=y.values, clf=clf)
#         plt.xlabel(feature1_name)
#         plt.ylabel(feature2_name)
#         plt.title(title)
     
i = 'ТУТ ОШИБКА GG'

# X = df[['habitat', 'odor']]
# y = df['class']
# X = pd.get_dummies(X)
# x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size=0.2)  
# print(X, y, sep='\n')


# knn_clf = Knn()
# knn_clf.fit(x_train, y_train)
# knn_clf_pred_res = knn_clf.predict(x_test)
# knn_clf_accuracy = accuracy_score(y_test, knn_clf_pred_res)

# print(f'KNN classifier accuracy: {knn_clf_accuracy:}')
# print(knn_clf_pred_res)

# tree = DecisionTreeRegressor(max_depth = 4)

# X = df[['habitat', 'odor', 'ring-number', 'cap-color', 'cap-surface']]
# y = df['class']
# X = pd.get_dummies(X)
# y = pd.get_dummies(y)

# x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size=0.2)  


# tree.fit(x_train, y_train)
# pred = tree.predict(x_test)

# cr = cross_val_score(tree, X, y )
# mse = mean_squared_error(y_test, pred)
# print(mse) #0.007256009582461658

# print(cr) #[0.99842445 0.99367111 0.9863042  0.76836356 0.97998254]



X = df[['habitat', 'odor', 'ring-number', 'cap-color', 'cap-surface']]
y = df['class']
X = pd.get_dummies(X)
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size=0.2)  


np.random.seed(42)  # For reproducibility
rf = ExtraTreesClassifier(n_estimators=100, random_state=5)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
cr = cross_val_score(rf, X, y )
mse = mean_squared_error(y_test, pred)
print(mse) #0.0043076923076923075

print(cr) #[1.         1.         0.99876923 0.97292308 0.95628079]




