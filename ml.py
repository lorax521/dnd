import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import numpy as np
import visualize
import generic
import matplotlib.pyplot as plt
import pandas as pd

def svm(df):
    # SVM
    y = df['wins'].values
    X = np.array([np.array(df['difficulty']), y]).T
    X = np.array(df['difficulty'])
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    # fit classifier to the training set
    model = SVC(gamma=2, C=1, probability=True)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print('ROC AUC: ' + str(roc_auc))

    if plot:
        visualize.plotModel(X, y, model)
    
    return model, df, model.score(X_test, y_test)


class svrModel:
    def __init__(self, df):
        self.df = df
        self.X = df['difficulty'].values
        self.y = df['wins'].values
        # Importing the dataset

        # Feature Scaling
        self.sc_X = StandardScaler()
        self.sc_y = StandardScaler()
        try:
            self.X = sc_X.fit_transform(X)
        except:
            self.X = self.X.reshape(-1, 1)
            self.X = self.sc_X.fit_transform(X)
        self.y_shape = self.y.reshape(len(self.y), 1)
        self.sc_y.fit_transform(self.y_shape)
        # self.y = self.sc_y.fit_transform(y_shape)

        # Fitting the SVR Model to the dataset
        # Create your model here
        self.model = SVR(kernel='rbf')
        self.model.fit(self.X, self.y)
    
    def predict(self, val):
        y_pred = self.sc_y.inverse_transform(self.model.predict(self.sc_X.transform(np.array([[val]]))))
        return y_pred
    
    def plot(self):
        # Visualising t`he SVR results
        X_grid = np.arange(min(self.X), max(self.X), 0.1)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(self.X, self.y, color = 'red')
        plt.plot(X_grid, self.model.predict(X_grid), color = 'blue')
        plt.title('Support Vector Regression')
        plt.xlabel('difficulty (independent)')
        plt.ylabel('wins (dependent)')
        plt.show()


m = svrModel(df)
m.predict(4)

df = df
X = df['difficulty'].values
y = df['wins'].values
# Importing the dataset

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
try:
    X = sc_X.fit_transform(X)
except:
    X = X.reshape(-1, 1)
    X = sc_X.fit_transform(X)
y_shape = y.reshape(len(y), 1)
# y = sc_y.fit_transform(y_shape)
sc_y.fit_transform(y_shape)

# Fitting the SVR Model to the dataset
# Create your model here
model = SVR(kernel='rbf')
model.fit(X, y)

y_pred = sc_y.inverse_transform(model.predict(sc_X.transform(np.array([[val]]))))

# Visualising t`he SVR results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, model.predict(X_grid), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('difficulty (independent)')
plt.ylabel('wins (dependent)')
plt.show()