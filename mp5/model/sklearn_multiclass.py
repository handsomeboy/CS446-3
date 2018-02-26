from sklearn import multiclass, svm
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC, SVC
import numpy as np

def sklearn_multiclass_prediction(mode, X_train, y_train, X_test):
    '''
    Use Scikit Learn built-in functions multiclass.OneVsRestClassifier
    and multiclass.OneVsOneClassifier to perform multiclass classification.

    Arguments:
        mode: one of 'ovr', 'ovo' or 'crammer'.
        X_train, X_test: numpy ndarray of training and test features.
        y_train: labels of training data, from 0 to 9.

    Returns:
        y_pred_train, y_pred_test: a tuple of 2 numpy ndarrays,
                                   being your prediction of labels on
                                   training and test data, from 0 to 9.
    '''
#     C = 1.0  # SVM regularization parameter
#     models = (svm.SVC(kernel='linear', C=C),
#           svm.LinearSVC(C=C),
#           svm.SVC(kernel='linear', multi_class='crammer_singer', C=C),
    
#     models = (clf.fit(X, y) for clf in models)
              
    if mode == 'ovr':
        clf = OneVsRestClassifier(LinearSVC(random_state=12345))
        clf.fit(X_train, y_train)
        
        y_pred_train = np.array(clf.predict(X_train))
        y_pred_test = np.array(clf.predict(X_test))
        
    elif mode == 'ovo':
        clf = OneVsOneClassifier(LinearSVC(random_state=12345))
        clf.fit(X_train, y_train)
        
        y_pred_train = np.array(clf.predict(X_train))
        y_pred_test = np.array(clf.predict(X_test))
        
    elif mode == 'crammer':
        clf = LinearSVC(random_state=12345,multi_class='crammer_singer')
        clf.fit(X_train, y_train)
        
        y_pred_train = np.array(clf.predict(X_train))
        y_pred_test = np.array(clf.predict(X_test))
    else:
        print("******* ATTENTION ********")
        print("Selected mode is not supported")
        print("Select from ovo, ovr, and crammer")
        
    return (y_pred_train, y_pred_test)
