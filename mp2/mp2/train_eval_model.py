"""
Train model and eval model helpers.
"""
from __future__ import print_function
import sys
import numpy as np
from models.linear_regression import LinearRegression


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=64,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    display_step = 100
    # Perform gradient descent.
#     x = np.array(processed_data[0]).astype(float)
#     y = np.array(processed_data[1]).astype(float)
#     batch_num = int(np.ceil(len(x)/batch_size))
    x = np.array(processed_dataset[0]).astype(float)
    y = np.array(processed_dataset[1]).astype(float)
    batch_num = int(np.ceil(len(x)/batch_size))
    for epoch in range(num_steps):
        
        #print(model.w)
        if shuffle:
            import random
            permutation = np.random.permutation(x.shape[0])
            x = np.array(x[permutation])
            y = np.array(y[permutation])
        for batch in range(batch_num):
            batch_start = batch * batch_size
            batch_end = (batch + 1 ) * batch_size 
            if batch+1 == batch_num:
                batch_end = len(x) - 1
            #print(batch_end)
            
            x_batch = np.array([x[i] for i in range(batch_start, batch_end)])
            y_batch = np.array([y[i] for i in range(batch_start, batch_end)])
            #print(y_batch.shape, x_batch.shape)
            update_step(x_batch, y_batch, model, learning_rate)
        if epoch % display_step == 0 :
                sys.stdout.write("\rProgress: {:2.1f}".format(100 * (epoch+1)/(num_steps)) \
                             + "% ... Training loss total: " + str(eval_model(processed_dataset, model)))
                sys.stdout.flush()   
         
            
        
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
    model.w = model.w - learning_rate * (total_grad)
    #print(total_grad)
    


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    y = np.array(processed_dataset[1]).astype(float)
    x = np.array(processed_dataset[0]).astype(float)
    f = model.forward(x)
    B = np.dot(np.transpose(model.x), model.x) 
    A = np.linalg.inv( B + model.w_decay_factor * np.eye(B.shape[1]))  
    model.w = np.dot(A, np.dot( np.transpose(model.x), y)) 
    print(model.w)
    return model.w


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    f = model.forward(processed_dataset[0])
    loss = model.total_loss(f, processed_dataset[1]) + model.w_decay_factor * np.sum(np.power(model.w,2))

    return loss
