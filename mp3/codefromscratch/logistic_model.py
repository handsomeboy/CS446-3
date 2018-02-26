"""logistic model class for binary classification."""

import numpy as np

class LogisticModel(object):
    
    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of W is the bias term, 
            self.W = [Bias, W1, W2, W3, ...] 
            where Wi correspnds to each feature dimension

        W_init needs to support:
          'zeros': initialize self.W with all zeros.
          'ones': initialze self.W with all ones.
          'uniform': initialize self.W with uniform random number between [0,1)
          'gaussian': initialize self.W with gaussion distribution (0, 0.1)

        Args:
            ndims(int): feature dimension
            W_init(str): types of initialization.
        """
        self.ndims = ndims
        self.W_init = W_init
        self.W = None
        if W_init == 'zeros':
            self.W = np.zeros(self.ndims+1)    
        elif W_init == 'ones':
            self.W = np.ones(self.ndims+1)
        elif W_init == 'uniform':
            self.W = np.random.uniform(0,1,self.ndims+1)
        elif W_init == 'gaussian':
            self.W = np.radnom.normal(0,0.1, self.ndims+1)
        else:
            print ('Unknown W_init ', W_init) 
        self.X = None
        
    def save_model(self, weight_file):
        """ Save well-trained weight into a binary file.
        Args:
            weight_file(str): binary file to save into.
        """
        self.W.astype('float32').tofile(weight_file)
        print ('model saved to', weight_file)

    def load_model(self, weight_file):
        """ Load pretrained weghit from a binary file.
        Args:
            weight_file(str): binary file to load from.
        """
        self.W = np.fromfile(weight_file, dtype=np.float32)
        print ('model loaded from', weight_file)

    def forward(self, X):
        """ Forward operation for logistic models.
            Performs the forward operation, and return probability score (sigmoid).
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): probability score of (label == +1) for each sample 
                             with a dimension of (# of samples,)
        """
        self.X = X
        f = 1./ ( 1. + np.exp(-np.matmul(self.X,self.W)))
        return f

    def backward(self, Y_true, X):
        """ Backward operation for logistic models. 
            Compute gradient according to the probability loss on lecture slides
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
        Returns:
            (numpy.ndarray): gradients of self.W
        """
        ###############################################################
        # Fill your code in this function
        
        Y_true.reshape((-1,1))
        
        D1 = -np.transpose(np.multiply(Y_true,np.transpose(X)))    
        D2 = np.exp(-np.multiply(Y_true,np.matmul(self.X,self.W)))
        N = np.add(1,D2)
        DN = np.divide(D2, N)       
        total_grad = np.matmul(DN, D1)
        
        return total_grad

    def classify(self, X):
        """ Performs binary classification on input dataset.
        Args:
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): predicted label = +1/-1 for each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        f = self.forward(X)
        
        return np.array([1  if f[i] >= 0.5 else -1 for i in range(len(f))])
    
    def fit(self, Y_true, X, learn_rate, max_iters):
        """ train model with input dataset using gradient descent. 
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            learn_rate: learning rate for gradient descent
            max_iters: maximal number of iterations
            ......: append as many arguments as you want
        """
        
        def accuracy(Y_t, Y_p):
            return np.sum([1 if Y_t[i] == Y_p[i] else 0 for i in range(len(Y_p))])/len(Y_p)
            
        for i in range(max_iters):
            total_grad = self.backward(Y_true, X)
            self.W -= learn_rate * total_grad
            '''
            if i% 100 == 0:
                Y_n = self.classify(X)
                acc = accuracy(Y_true, Y_n)
                #print(acc)
            '''
        return self.W
            
            
        
        

    