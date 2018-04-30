"""Generative adversarial network."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers

class Gan(object):
    """Adversary based generator network.
    """
    def __init__(self, ndims=784, nlatent=2):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """
        tf.reset_default_graph()
        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, self._ndims])

        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, self._nlatent])
        
        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)
        y_hat = self._discriminator(self.x_hat)
        y = self._discriminator(self.x_placeholder, reuse=True)
        

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)

        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Add optimizers for appropriate variables
        self.lr_placeholder = tf.placeholder(tf.float32, [])
        
        self.model_param = tf.trainable_variables()
        self.g_param = [p for p in self.model_param if p.name.startswith('generator')]
        self.d_param = [p for p in self.model_param if p.name.startswith('discriminator')]
        
        
        self.g_train_opt = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)\
                                        .minimize(self.g_loss, var_list=self.g_param)
        self.d_train_opt = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder)\
                                        .minimize(self.d_loss, var_list=self.d_param)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        


    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1). 
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            layer_1 = layers.fully_connected(x,392,activation_fn=None)
            layer_1 = tf.maximum(0.01*layer_1, layer_1)
            layer_2 = layers.fully_connected(layer_1,196 , activation_fn=None)
            layer_2 = tf.maximum(0.01*layer_2, layer_2)
            layer_3 = layers.fully_connected(layer_2, 1, activation_fn=None)
            
            return layer_3


    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=y,
                                                    labels=tf.ones_like(y)))
        loss_fake = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, 
                                                          labels=tf.zeros_like(y_hat)))
        l = loss_real + loss_fake
        return l


    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation 
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:
            layer_1 = layers.fully_connected(z, 256, activation_fn=None)
            layer_1 = tf.maximum(0.01*layer_1, layer_1)
            layer_2 = layers.fully_connected(layer_1, 400, activation_fn=None)
            layer_2 = tf.maximum(0.01*layer_2, layer_2)
            x_hat = layers.fully_connected(layer_2, self._ndims, activation_fn=tf.nn.tanh)
    
            return x_hat


    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        l = tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat,
                                                     labels=tf.ones_like(y_hat)))
        return l
  
