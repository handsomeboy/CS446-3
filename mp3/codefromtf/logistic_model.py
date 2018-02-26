"""logistic model class for binary classification."""
import tensorflow as tf
import numpy as np

class LogisticModel_TF(object):
    
    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of Weight is the bias term, 
            Weight = [Bias, W1, W2, W3, ...] 
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
        self.W0 = None
        ###############################################################
        # Fill your code below
        ###############################################################
        if W_init == 'zeros':
            # Hint: self.W0 = tf.zeros([self.ndims+1,1])
            self.W0 = tf.zeros([self.ndims+1, 1])
        elif W_init == 'ones':
            self.W0 = tf.ones([self.ndims+1, 1])
        elif W_init == 'uniform':
            self.W0 = tf.random_uniform([self.ndims+1, 1], maxval=1)
        elif W_init == 'gaussian':
            self.W0 = tf.random_normal([self.ndims+1, 1],mean=0.0,stddev=0.1)
        else:
            print ('Unknown W_init ', W_init) 
        #self.graph = tf.Graph()
        
    def build_graph(self, learn_rate):
        """ build tensorflow training graph for logistic model.
        Args:
            learn_rate: learn rate for gradient descent
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        # Hint: self.W = tf.Variable(self.W0)
        self.W = tf.Variable(self.W0)
        self.lr = learn_rate
#         self.X_TF = X 
#         self.y_TF = np.array([Y_true]).T.astype(int) 
        self.X_TF = tf.placeholder(tf.float32, [None, self.ndims+1])
        self.y_TF = tf.placeholder(tf.int32, [None, 1])
        self.predictions = tf.sigmoid(tf.matmul(tf.cast(self.X_TF,tf.float32), self.W))
        self.cost = tf.reduce_mean(tf.square(tf.subtract(tf.cast(self.y_TF,tf.float32), self.predictions)))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.cost)        

        pass
        
    def fit(self, Y_true, X, max_iters):
        """ train model with input dataset using gradient descent. 
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of (# of samples,1)
            X(numpy.ndarray): input dataset with a dimension of (# of samples, ndims+1)
            max_iters: maximal number of training iterations
            ......: append as many arguments as you want
        Returns:
            (numpy.ndarray): sigmoid output from well trained logistic model, used for classification
                             with a dimension of (# of samples, 1)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        def accuracy(Y_t, Y_p):
            acc_vec = np.array([1 if Y_t[i] == Y_p[i] else 0 for i in range(len(Y_p))])
            acc_val = np.mean(acc_vec)
            return acc_vec, acc_val
        
#         self.build_graph(learn_rate, Y_true, X)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            Y = np.array([Y_true]).T.astype(int)    
            for epoch in range(max_iters):
                sess.run(self.optimizer, feed_dict={self.X_TF: X, self.y_TF: Y})
                if epoch % 200 == 0:
                    pred = sess.run(self.predictions, feed_dict={self.X_TF: X})
                    self.classify = np.array([1  if pred[i]>= 0.5 else 0 for i in range(len(pred))])
                    acc_vec, acc_val = accuracy(Y, self.classify)
                    cost = sess.run(self.cost, feed_dict={self.X_TF: X, self.y_TF: Y})
                    
                    #print(epoch, '..', cost,acc_val)
                if epoch+1 == max_iters:
                    pred = sess.run(self.predictions, feed_dict={self.X_TF: X})
                    self.classify = np.array([1  if pred[i]>= 0.5 else 0 for i in range(len(pred))])
                    acc_vec, acc_val = accuracy(Y, self.classify)
                    #print("Final step accuracy:", acc_val)
                    return pred
                    
    
    