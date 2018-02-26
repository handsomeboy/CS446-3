import tensorflow as tf


def toy_fn_1(arg1, arg2):
    '''Given two tensors of arbitrary (but same) rank and size, build a computation
    graph for the following function, which should be computed element-wise:

    arg1^3 + 4*arg2^2 - 10*arg1

    Args:
        arg1(tf.Tensor): A tensor of arbitrary rank
        arg2(tf.Tensor): A tensor of the same rank as arg1
    Returns:
        (tf.Tensor): the result of the computation (same rank as inputs)
    '''
    # Input your code here
    T1 = tf.Variable(arg1)
    T2 = tf.Variable(arg2)
    init = tf.global_variables_initializer()   
        
    with tf.Session() as sess:
        sess.run(init)
        return sess.run(tf.add(tf.add(tf.pow(T1, 3), 4 * tf.pow(T2, 2)), -10 * T1))


def toy_fn_2(arg1, arg2):
    '''Given a rank-two tensor and a rank-one tensor, build a computation graph
    that computes the following:

    first, it sums over the first dimension of the rank-two tensor
    (zero-indexed - i.e. sum over the rows). It then subtracts the maximum
    value of the rank-1 tensor from each element of the result.

    Args:
        arg1(tf.Tensor): A rank-2 tensor with dimensions (m, n)
        arg2(tf.Tensor): A rank-1 tensor with dimension p
    Returns:
        (tf.Tensor): the result of the computation, which is a rank-1 tensor
          with dimension m
    '''
    # Input your code here
    T1 = tf.Variable(arg1)
    T2 = tf.Variable(arg2)
    init = tf.global_variables_initializer()  
    
    with tf.Session() as sess:
        sess.run(init)
        return sess.run(tf.add(tf.reduce_sum(T1, axis=1), -1 * tf.ones_like(tf.reduce_sum(T1, axis=1)) * tf.reduce_max(T2) ))


def toy_fn_3(arg1, arg2):
    '''
    Given two rank-one tensors of the same size, build a computation graph that
    builds a rank-one tensor by interleaving the two original tensors. For
    example, given the following inputs:

    arg1 = [1, 2]
    arg2 = [10, 20]

    The result should be [1, 10, 2, 20]

    Hint: this can be accomplished by first creating a rank-two tensor whose
    columns are the two original tensors and then reshaping it. Make sure the
    final tensor is rank-1!

    Args:
        arg1(tf.Tensor): A rank-1 tensor with dimension m
        arg2(tf.Tensor): A rank-1 tensor with dimension m
    Returns:
        (tf.Tensor): the result of the computation, which is a rank-1 tensor
          with dimension 2*m
    '''
    # Input your code here
    output = tf.Variable([arg1[int(i/2)] if i % 2 ==0 else arg2[int((i-1)/2)] for i in range(arg1.shape[0]+arg2.shape[0])])
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()  )        
        return sess.run(output)
