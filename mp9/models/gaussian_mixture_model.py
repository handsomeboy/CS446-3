"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""
    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=1e-6):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters
        self._mu = np.zeros((n_components, n_dims)) # np.array of size (n_components, n_dims)

        # Initialized with uniform distribution.
        self._pi = np.array([1.0/n_components for i in range(n_components)]).reshape((-1,1))# np.array of size (n_components, 1)

        # Initialized with identity.
        self._sigma = np.array([1000*np.eye(n_dims) for i in range(n_components)])  # np.array of size (n_components, n_dims, n_dims)

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        if self._mu.all() == 0:
            l_rand = np.random.choice(range(len(x)), self._n_components, replace=False)
            self._mu = x[l_rand]
            print(l_rand)
        for i in range(self._max_iter):
            print("iteration "+str(i))
            z_ik = self._e_step(x)
            self._m_step(x, z_ik)
            
        
    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        
        return self.get_posterior(x) 

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        self._pi = np.sum(z_ik, axis=0)/len(x)
        for k in range(self._n_components):
            self._mu[k] = np.matmul(z_ik[:,k],x)/(len(x)*self._pi[k])
            self._sigma[k] = np.cov((x-self._mu[k]).T, aweights=z_ik[:,k], ddof=0, bias=True)
        pass

    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N,, n_components).
        """
        ret = np.zeros((len(x), self._n_components))
        for k in range(self._n_components):
            ret[:,k] = self._multivariate_gaussian(x, self._mu[k], self._sigma[k])
        return ret

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        marginal = np.array((len(x),))
        marginal = np.matmul(self.get_conditional(x), self._pi).reshape((-1,))
        return marginal 

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        z_ik = np.zeros((len(x), self._n_components))
        p_cond = self.get_conditional(x)
        for k in range(self._n_components):
            z_ik[:,k] =(p_cond[:,k]*self._pi[k] + self._reg_covar) / (self.get_marginals(x)+ self._n_components*self._reg_covar)

            if np.sum(z_ik[:,k]) == 0:
                print("error")
        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, ndims)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        self.cluster_label_map = []
        p = self.get_posterior(x)
        data_to_cluster = np.argmax(p, axis=1)
        label = np.zeros((len(np.unique(y)),self._n_components))
        unique_label = np.unique(y).tolist()
        
        for i in range(len(x)):
            labelindx = unique_label.index(y[i])
            label[labelindx,data_to_cluster[i]] +=1
        convertor = np.argmax(label, axis=0)
        for i in range(len(convertor)):
            self.cluster_label_map.append(unique_label[convertor[i]])
        return self.cluster_label_map

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """

        z_ik = self.get_posterior(x)
        label = np.argmax(z_ik, axis=1)
        y_hat =[]
        for i in range(len(x)):
            y_hat.append(self.cluster_label_map[label[i]])

        return np.array(y_hat)
