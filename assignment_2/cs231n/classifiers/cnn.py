import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

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
    for i in range(1,4):
        self.params['W'+str(i)] = weight_scale*np.random.randn(num_filters,weight_depth[i-1],filter_size,filter_size)
        self.params['b'+str(i)] = np.zeros((num_filters))

    
    self.params['fc'] = weight_scale*np.random.randn(hidden_dim, num_filters, num_filters/8,num_filters/8)
    self.params['class_weight'] = weight_scale*np.random.randn(num_classes,hidden_dim,1,1)
    self.params['b_fc']         = np.zeros((hidden_dim))
    self.params['b_class_weight'] = np.zeros((num_classes)) 


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
    for i in range(1,4):
        w  = self.params['W'+str(i)]
        b  = self.params['b'+str(i)]
        store_ouput[str(i)], store_cache[str(i)] = conv_relu_pool_forward(store_ouput[str(i-1)], w, b, conv_param, pool_param)

    # print store_ouput[str(i)].shape, self.params['fc'].shape
    

    store_ouput[str(i+1)], store_cache[str(i+1)] = conv_forward_fast(store_ouput[str(i)], self.params['fc'], self.params['b_fc'], conv_fc_param)

    scores, store_cache[str(i+2)] = conv_forward_fast(store_ouput[str(i+1)], self.params['class_weight'], self.params['b_class_weight'], conv_fc_param)

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
    for i in range(1,4):
        loss_reg += np.sum(self.params['W'+str(i)]**2.0)*self.reg*0.5
    loss_reg += 0.5*self.reg*(np.sum(self.params['fc']**2.0)+ np.sum(self.params['class_weight']**2.0))
    loss += loss_reg

    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    dx, grads['class_weight'], grads['b_class_weight'] = conv_backward_fast(dscores, store_cache[str(5)])
    dx, grads['fc'], grads['b_fc'] = conv_backward_fast(dx, store_cache[str(4)])
    for i in range (3, 0, -1):
        dx, grads['W' + str(i)], grads['b' + str(i)] = conv_relu_pool_backward(dx, store_cache[str(i)])
    
    grads['class_weight']   += self.reg*self.params['class_weight']
    grads['fc']             += self.reg*self.params['fc']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

