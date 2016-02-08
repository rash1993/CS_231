import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    weight_depth = [input_dim[0], num_filters, num_filters]
    self.params['W1'] = weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
    self.params['b1'] = np.zeros((num_filters))


    self.params['W2'] = weight_scale*np.random.randn(hidden_dim, num_filters, input_dim[1]/2,input_dim[2]/2)
    self.params['W3'] = weight_scale*np.random.randn(num_classes,hidden_dim,1,1)
    self.params['b2'] = np.zeros((hidden_dim))
    self.params['b3'] = np.zeros((num_classes))


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    for i in self.params:
        print i, self.params[i].shape


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    conv_fc_param = {'stride': 1, 'pad': 0}
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    store_cache = {}
    store_ouput = {}
    store_ouput['0'] = X
    store_ouput['1'], store_cache[str(1)] = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

    # print store_ouput[str(i)].shape, self.params['fc'].shape


    store_ouput[str(2)], store_cache[str(2)] = conv_forward_fast(store_ouput['1'], W2, b2, conv_fc_param)

    scores, store_cache['3'] = conv_forward_fast(store_ouput['2'], W3, b3, conv_fc_param)

    # print scores.shape
    # store_ouput[str(i+1)], store_cache[str(i+1)] = affine_forward(store_ouput[str(i)], self.params['fc'],0)

    # store_ouput[str(i+1)] = np.inner(store_ouput[str(i)], self.params['fc'].T)

    # print store_ouput[str(i+1)].shape, self.params['class_weight'].shape
    # scores, store_cache[str(i+2)] = affine_forward(store_ouput[str(i+1)], self.params['class_weight'],0)
    # scores = np.inner(store_ouput[str(i+1)], self.params['class_weight'])




    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    loss, grads = 0, {}

    if y is None:
      return scores
    loss, dscores = softmax_loss(scores,y)
    loss_reg = 0.0

    loss_reg += (np.sum(W1**2.0)+np.sum(W2**2.)+np.sum(W3**2.))*self.reg*0.5
    loss += loss_reg

    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    dx, grads['W3'], grads['b3'] = conv_backward_fast(dscores, store_cache[str(3)])
    dx, grads['W2'], grads['b2'] = conv_backward_fast(dx, store_cache[str(2)])
    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, store_cache[str(1)])

    grads['W1']   += self.reg*self.params['W1']
    grads['W2']   += self.reg*self.params['W2']
    grads['W3']   += self.reg*self.params['W3']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
