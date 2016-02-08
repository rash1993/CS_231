import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm = False):

        C, H, W = input_dim
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.bn_params = {}
        self.reg = reg
        self.dtype = dtype

        # Conv layer
        # The parameters of the conv is of size (F,C,HH,WW) with
        # F give the nb of filters, C,HH,WW characterize the size of
        # each filter
        # Input size : (N,C,H,W)
        # Output size : (N,F,Hc,Wc)
        F = num_filters
        filter_height = filter_size
        filter_width = filter_size
        stride_conv = 1  # stride
        P = (filter_size - 1) / 2  # padd
        Hc = (H + 2 * P - filter_height) / stride_conv + 1
        Wc = (W + 2 * P - filter_width) / stride_conv + 1

        W1 = weight_scale * np.random.randn(F, C, filter_height, filter_width)
        b1 = np.zeros(F)

        # Pool layer : 2*2
        # The pool layer has no parameters but is important in the
        # count of dimension.
        # Input : (N,F,Hc,Wc)
        # Ouput : (N,F,Hp,Wp)

        width_pool = 2
        height_pool = 2
        stride_pool = 2
        Hp = (Hc - height_pool) / stride_pool + 1
        Wp = (Wc - width_pool) / stride_pool + 1

        # Hidden Affine layer
        # Size of the parameter (F*Hp*Wp,H1)
        # Input: (N,F*Hp*Wp)
        # Output: (N,Hh)

        Hh = hidden_dim
        W2 = weight_scale * np.random.randn(F * Hp * Wp, Hh)
        b2 = np.zeros(Hh)

        # Output affine layer
        # Size of the parameter (Hh,Hc)
        # Input: (N,Hh)
        # Output: (N,Hc)

        Hc = num_classes
        W3 = weight_scale * np.random.randn(Hh, Hc)
        b3 = np.zeros(Hc)
        self.params = {}
        self.params.update({'W1': W1,
                            'W2': W2,
                            'W3': W3,
                            'b1': b1,
                            'b2': b2,
                            'b3': b3})

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.

        if self.use_batchnorm:
            print 'We use batchnorm here'
            bn_param1 = {'mode': 'train',
                         'running_mean': np.zeros(F),
                         'running_var': np.zeros(F)}
            gamma1 = np.ones(F)
            beta1 = np.zeros(F)

            bn_param2 = {'mode': 'train',
                         'running_mean': np.zeros(Hh),
                         'running_var': np.zeros(Hh)}
            gamma2 = np.ones(Hh)
            beta2 = np.zeros(Hh)

            self.bn_params.update({'bn_param1': bn_param1,
                                   'bn_param2': bn_param2})

            self.params.update({'beta1': beta1,
                                'beta2': beta2,
                                'gamma1': gamma1,
                                'gamma2': gamma2})

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)


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
    N, F, Hp, Wp = store_ouput['1'].shape
    store_ouput['1'] = store_ouput['1'].reshape(X.shape[0],-1)
    store_ouput[str(2)], store_cache[str(2)] = affine_relu_forward(store_ouput['1'], W2, b2)

    scores, store_cache['3'] = affine_forward(store_ouput['2'], W3, b3)

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


    dx, grads['W3'], grads['b3'] = affine_backward(dscores, store_cache[str(3)])
    dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, store_cache[str(2)])
    dx = dx.reshape(N, F, Hp, Wp)
    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, store_cache[str(1)])

    grads['W1']   += self.reg*self.params['W1']
    grads['W2']   += self.reg*self.params['W2']
    grads['W3']   += self.reg*self.params['W3']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
