import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
import visualize
import generic

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