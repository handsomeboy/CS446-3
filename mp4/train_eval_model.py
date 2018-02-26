"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)

    x = data['image']
    y = data['label'].reshape(-1,1)
    #print("shape of y  :", y.shape)
    batch_num = int(np.ceil(len(x)/batch_size))
    for epoch in range(num_steps):
        if shuffle:
            import random
            permutation = np.random.permutation(x.shape[0])
            x = np.array(x[permutation])
            y = np.array(y[permutation])
        for batch in range(batch_num):
            batch_start = batch * batch_size
            batch_end = (batch + 1 ) * batch_size
            if batch+1 == batch_num:
                batch_end = len(x)
            x_batch = np.array([x[i] for i in range(batch_start, batch_end)])
            y_batch = y[batch_start:batch_end]
            #print(y_batch.shape)

            update_step(x_batch, y_batch, model, learning_rate)
        f = model.forward(x)
        print(model.total_loss(f, y))


    return model



def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    f = model.forward(x_batch)
    total_grad = model.backward(f, y_batch)
    model.w = model.w -learning_rate * total_grad

def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    #print("q",q.shape)
    #print("P",P)
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)

    # Set model.w
    model.w = z[0:model.dims+1]
    print(model.w)


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    x = data['image']
    y = data['label']
    model.forward(x)
    z_len = model.ndims + 1+ len(model.x)
    #print(z_len)
    P = self.w_decay_factor*np.eye(z_len)
    q = np.zeros((z_len,1))
    h = -np.ones((2*len(y),1))

    for i in range(model.ndims+1,z_len):
        P[i,i] = 0
        q[i] = -1
        h[i-self.ndims+1+len(y)] = 0
     
    G = np.zeros((2*len(y),z_len)
    G[0:len(y),0:model.ndims+1] = - y * model.x
    for i in range(len(y)):
        G[i+len(y), i+self.ndims+1] = -1
        G[i,i+model.ndims+1] = -1
    #print("q: " , q.shape)
    #print("P: ", P.shape)
    #h = -np.ones((model.ndims+1+len(x),1))
    # Implementation here.
    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    f = model.forward(data['image'])
    pred = model.predict(f).astype(int)
    y = data['label'].astype(int)
    loss = model.total_loss(f, y) + model.w_decay_factor * np.sum(np.power(model.w,2))
    acc = np.sum([1 if y[i] == pred[i] else 0  for i in range(len(y))]) /len(y)
    return loss, acc
