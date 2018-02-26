from sklearn import multiclass, svm
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC, SVC
import numpy as np

class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        ovr_dict ={}
        n = len(self.labels)
        for i in range(n):
            clf = LinearSVC(random_state=12345)
            y_b = np.array([1 if y[cnt]==i else 0 for cnt in range(len(y))]).reshape(-1,1)
            clf.fit(X, y_b)
            ovr_dict.update({i: clf})
        
        return ovr_dict

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        ovo_dict = {}
        n = len(self.labels)
        for i in range(n):
            for j in range(i+1,n):
                clf = LinearSVC(random_state=12345)
                y_b = np.array([])
                X_b = np.array([])
        
                for cnt in range(len(y)):
                    if y[cnt] == i:
                        y_b = np.append(y_b, [1])
                        X_b = np.append(X_b, X[cnt])
                    elif y[cnt] == j:
                        y_b = np.append(y_b, [0])
                        X_b = np.append(X_b, X[cnt])
                y_b = y_b.reshape(-1,1)    
                X_b = X_b.reshape(-1,len(X[0]))
                #print(X_b.shape, y_b.shape)
                clf.fit(X_b, y_b)         
                ovo_dict.update({(i,j): clf})
                
        return  ovo_dict

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        n_label = len(self.labels)
        n_sample = X.shape[0]
        output = np.zeros((n_sample, n_label))
        
        for i in self.binary_svm:
            #print(output.shape)
            #print(self.binary_svm[i].decision_function(X).shape)
            output[:, i] = self.binary_svm[i].decision_function(X)
            
        
        return output                 

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        ''' 
        n_label = len(self.labels)
        n_sample = len(X)
        output = np.zeros((n_sample, n_label))
        for j in range(n_label):
            for k in range(j+1, n_label):
                output[:,j] += self.binary_svm[(j,k)].predict(X)
                output[:,k] += np.ones_like(self.binary_svm[(j,k)].predict(X)) - self.binary_svm[(j,k)].predict(X)
        
        return output    

    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        K = len(W)
        d = len(W[0])
        N = len(X)
        
        delta_j_i = np.zeros((N,K))
        loss_i = 0
        for i in range(N):
            delta_j_i[i][y[i]] = 1
            
        loss_nm = np.dot(X, W.T) + np.ones((N,K)) - delta_j_i
        
        idx = np.argmax(loss_nm, axis=1)
        
        a = np.array([i for i in range(N)])

        loss = np.sum(loss_nm[a,idx])
        
        loss -= np.sum([np.dot(W[y[i]],X[i]) for i in range(len(X))])
        
        
        return C*loss + 0.5*np.sum(power(W,2))
    
    

    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
         '''
        K = len(W)
        d = len(W[0])
        N = len(X)
        
        delta_j_i = np.zeros((N,K))
        
        for i in range(N):
            delta_j_i[i][y[i]] = 1
            
        loss_nm = np.dot(X, W.T) + np.ones((N,K)) - delta_j_i
        
        idx = np.argmax(loss_nm, axis=1)
        
        a = np.array([i for i in range(N)])
        
        grad = np.zeros((K,d))
        for i in range(N):
            grad[idx[i], :] += C * X[i].T
            grad[y[i],:] -= C*X[i].T
        grad +=W

        
        return grad 
