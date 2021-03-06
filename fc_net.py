import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.

  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """

  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################

    self.params['W1'] = np.random.normal(0,weight_scale,(input_dim,hidden_dim))
    self.params['W2'] = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
    self.params['b1'] = np.zeros((hidden_dim))
    self.params['b2'] = np.zeros((num_classes))

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters
    """
    scores = None
    W1 = self.params['W1']
    W2 = self.params['W2']
    b1 = self.params['b1'].reshape(W1.shape[1],1)
    b2 = self.params['b2'].reshape(W2.shape[1],1)

    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################

    score1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])

    scores, cache2 = affine_forward(score1,self.params['W2'],self.params['b2'])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    loss, dscores = softmax_loss(scores,y)
    dscores1, dW2, db2 = affine_backward(dscores, cache2)
    dx, dW1, db1 = affine_relu_backward(dscores1,cache1)
    loss += 0.5*(sum(np.diagonal(np.dot(W1,W1.T)))+         sum(np.diagonal(np.dot(W2,W2.T))))*self.reg
    grads['W1'] = dW1 + self.reg*W1
    grads['W2'] = dW2 + self.reg*W2
    grads['b1'] = db1 #+ self.reg*b1 #Stupid, you thought to regularize the bias
    grads['b2'] = db2 #+ self.reg*b2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be

  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.

  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.

    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    self.num_hidden_layer = len(hidden_dims)
    dim1 = input_dim
    for i,hidden_size in enumerate(hidden_dims):
        self.params["W"+str(i+1)] = np.random.normal(0,weight_scale,(dim1,hidden_size))#choose better initialization scheme
        self.params["b"+str(i+1)] = np.zeros((hidden_size))
        dim1 = hidden_size

    self.params["W"+str(i+2)] = np.random.normal(0,weight_scale,(dim1,num_classes))
    self.params["b"+str(i+2)] = np.zeros((num_classes))
    ############################################################################
    ###############Initialize batch_norm parameters#############################

    if self.use_batchnorm:
        for i in range(1,self.num_hidden_layer+1):
            self.params['gamma'+str(i)] = np.random.normal(0,weight_scale,(self.params["W"+str(i)].shape[1]))
            self.params['beta'+str(i)] = np.random.normal(0,weight_scale,(self.params["W"+str(i)].shape[1]))
    # print hidden_dims
    # for k in self.params:
        # print k, self.params[k].shape


    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if (self.use_batchnorm):
        self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    # Cast all parameters to the correct datatype

    for k, v in self.params.iteritems():
        self.params[k] = v.astype(dtype)



  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    # score1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])

    if self.use_dropout>0:
        if y == None:
            scores = af_rl_dp(self, X, y, mode)
            return scores#*self.dropout_param['p']
        else:
            loss, grads = af_rl_dp(self, X, y, mode)
    elif self.use_batchnorm:
        if y==None:
            scores = af_rl_bn(self, X, y, mode)
            return scores
        else:
            loss, grads = af_rl_bn(self, X, y, mode)
    else:
        if y==None:
            scores = af_rl(self, X, y, mode)
            return scores
        else:
            loss, grads = af_rl(self, X, y, mode)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    # print mode

    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    return loss, grads
    ############################################################################

def af_rl_dp_forward (x, w, b, dropout_param):
    '''It calculates the forward pass for the affine+relu+dropout layer
    '''
    out, cache_affine  = affine_forward(x,w,b)
    out, cache_relu    = relu_forward(out)
    out, cache_dropout = dropout_forward(out, dropout_param)

    cache = (cache_affine, cache_relu, cache_dropout)
    return out, cache

def af_rl_dp_backward(dout, cache):
    '''
    '''
    cache_affine, cache_relu, cache_dropout,= cache

    dx = relu_backward(dout, cache_relu)
    dx = dropout_backward(dx, cache_dropout)
    dx, dw, db = affine_backward(dx, cache_affine)

    return dx, dw, db


def af_rl_bn_forward(x, w, b, gamma, beta, bn_param):
  '''It caculates the forward pass for affine+relu+batch_norm layers
  '''
  out, cache_affine = affine_forward(x,w,b)
  out, cache_bnorm  = batchnorm_forward(out,gamma, beta, bn_param)
  out, cache_relu   = relu_forward(out)

  cache = (cache_affine, cache_bnorm, cache_relu)

  return out, cache

def af_rl_bn_backward(dout, cache):
  '''It caculates the backward pass for affine+relu+batch_norm layers
  '''
  cache_affine, cache_bnorm, cache_relu = cache

  dx = relu_backward(dout, cache_relu)
  dx, dgamma, dbeta = batchnorm_backward_alt(dx, cache_bnorm)
  dx, dw, db = affine_backward(dx, cache_affine)

  return dx, dw, db, dgamma, dbeta



def af_rl(self, X, y, mode):
    '''The function to compute the loss and grads when there are only two layers
    affine + relu present
    '''
    store_cache = {}
    store_ouput = {}
    store_ouput['0'] = X

    for i in range(1,self.num_hidden_layer+1):
        w  = self.params['W'+str(i)]
        b  = self.params['b'+str(i)]
        store_ouput[str(i)], store_cache[str(i)] = affine_relu_forward(store_ouput[str(i-1)], w, b)


    scores, store_cache[str(i+1)] = affine_forward(store_ouput[str(i)],self.params['W'+str(i+1)],self.params['b'+str(i+1)])

    if y==None:
      return scores

    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    loss, dscores = softmax_loss(scores,y)
    for i in range(1,self.num_hidden_layer+2):
        loss += 0.5*self.reg*((self.params['W'+str(i)]**2).sum())

    #####################Calculate Grads####################################
    dout, grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = affine_backward(dscores,store_cache[str(self.num_layers)])

    for i in range(self.num_hidden_layer,0,-1):
        dout, grads['W'+str(i)], grads['b'+str(i)] = affine_relu_backward(dout, store_cache[str(i)])

    for i in range(1,self.num_hidden_layer+2):
        grads['W'+str(i)] += self.params['W'+str(i)]*self.reg
    ######################################################################
    return loss, grads


def af_rl_bn (self, X, y, mode):
    store_cache = {}
    store_ouput = {}
    store_ouput['0'] = X

    for i in range(1,self.num_hidden_layer+1):
        w     = self.params['W'+str(i)]
        b     = self.params['b'+str(i)]
        gamma = self.params['gamma'+str(i)]
        beta  = self.params['beta'+str(i)]
        bn_param = self.bn_params[i-1]
        store_ouput[str(i)], store_cache[str(i)] = af_rl_bn_forward(store_ouput[str(i-1)], w, b, gamma, beta, bn_param)


    scores, store_cache[str(i+1)] = affine_forward(store_ouput[str(i)],self.params['W'+str(i+1)],self.params['b'+str(i+1)])
    if mode == 'test':
      return scores

    if y==None:
        return scores

    loss, grads = 0.0, {}
    loss, dscores = softmax_loss(scores,y)

    for i in range(1,self.num_hidden_layer+2):
        loss += 0.5*self.reg*((self.params['W'+str(i)]*self.params['W'+str(i)]).sum())

    # for i in range(1,self.num_hidden_layer+1):
        # loss += 0.5*self.reg*((self.params['gamma'+str(i)]**2).sum())#+(self.params['beta'+str(i)]**2).sum())

    ##########################Calculate the Grads###########################
    dout, grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = affine_backward (dscores,store_cache[str(self.num_layers)])

    for i in range(self.num_hidden_layer,0,-1):
        dout, grads['W'+str(i)], grads['b'+str(i)], grads['gamma'+str(i)], grads['beta'+str(i)] = af_rl_bn_backward(dout, store_cache[str(i)])

    for i in range(1,self.num_hidden_layer+2):
        grads['W'+str(i)] += self.params['W'+str(i)]*self.reg

    # for i in range(1,self.num_hidden_layer+1):
            # grads['gamma'+str(i)] += self.reg*self.params['gamma'+str(i)]
            #grads['beta'+str(i)]  += self.reg*self.params['beta'+str(i)]
    return loss, grads

def af_rl_dp (self, X, y, mode):
    store_cache = {}
    store_ouput = {}
    store_ouput['0'] = X
    for i in range(1,self.num_hidden_layer+1):
        w     = self.params['W'+str(i)]
        b     = self.params['b'+str(i)]
        store_ouput[str(i)], store_cache[str(i)] = af_rl_dp_forward (store_ouput[str(i-1)], w, b, self.dropout_param)
    scores, store_cache[str(i+1)] = affine_forward(store_ouput[str(i)],self.params['W'+str(i+1)],self.params['b'+str(i+1)])

    if mode == 'test':
      return scores

    if y==None:
        return scores

    loss, grads = 0.0, {}
    loss, dscores = softmax_loss(scores,y)
    for i in range(1,self.num_hidden_layer+2):
        loss += 0.5*self.reg*((self.params['W'+str(i)]**2).sum())

    #####################Calculate Grads####################################
    dout, grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = affine_backward(dscores,store_cache[str(self.num_layers)])

    for i in range(self.num_hidden_layer,0,-1):
        dout, grads['W'+str(i)], grads['b'+str(i)] = af_rl_dp_backward (dout, store_cache[str(i)])

    for i in range(1,self.num_hidden_layer+2):
        grads['W'+str(i)] += self.params['W'+str(i)]*self.reg
    ######################################################################

    return loss, grads
