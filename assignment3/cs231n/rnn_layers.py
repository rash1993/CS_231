import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  next_h = np.tanh(np.dot(x,Wx)+np.dot(prev_h,Wh)+b)
  # print next_h.shape
  cache = (x, prev_h, Wx, Wh, b, next_h)
  # print next_h.shape
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward (dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.

  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass

  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (D, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  (x, prev_h, Wx, Wh, b, next_h) = cache

  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################

  sec2_h = (1-next_h**2)
  dWh = np.dot(prev_h.T, (sec2_h*dnext_h) )

  dWx = np.dot(x.T, sec2_h * dnext_h)
  db  = np.sum(sec2_h*dnext_h, axis=0)

  dprev_h = np.dot(sec2_h * dnext_h, Wh.T)
  dx = np.dot(sec2_h*dnext_h, Wx.T)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.

  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  N, T, D = x.shape
  N, H = h0.shape
  cache = {}
  h = np.zeros((N, T, H),dtype = np.float64)
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  h[:,0,:], cache[0] = rnn_step_forward(x[:,0,:], h0, Wx, Wh, b)
  for t in range(T-1):
      h[:,t+1, :], cache[t+1] = rnn_step_forward(x[:,t+1,:], h[:, t, :], Wx, Wh, b)
    #   print h[:,t+1, :]

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.

  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)

  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  (N, T, H) = dh.shape
  (x, prev_h, Wx, Wh, b, next_h) = cache[T-1]
  N, D = x.shape
  dx = np.zeros((N,T,D), dtype = np.float32)
  dWx = np.zeros((D, H), dtype = np.float32)
  dWh = np.zeros((H, H), dtype = np.float32)
  db = np.zeros((H), dtype = np.float32)

  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  dx[:,T-1,:], p, DWx, DWh, Db = rnn_step_backward(dh[:, T-1, :],  cache[T-1])
  dWx = dWx + DWx
  dWh += DWh
  db  += Db
  for t in range(T-2,-1,-1):
    grad_h = p + dh[:,t,:]
    dx[:,t,:], p, DWx, DWh, Db = rnn_step_backward(grad_h, cache[t])
    dWx = dWx + DWx
    dWh += DWh
    db  += Db
  # print dWx.shape, dWh.shape, db.shape
  dh0 = p
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.

  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.

  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  N,T = x.shape
  V, D = W.shape
  temp = np.zeros((N, T, V))

  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  # n = range(N)
  # t = range(T)
  # temp[n,t,x[n,t]]
  for n in range (N):
    for t in range (T):
        temp[n, t, x[n,t]] = 1
  # np.add.at(temp,(n,t,x[n,t]), 1)
  out = np.dot(temp, W)
  cache = (out,temp)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.

  HINT: Look up the function np.add.at

  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass

  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  (out,temp) = cache
  N,T,V = temp.shape
  _,_,D = dout.shape
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  dW = np.dot(np.transpose(temp, (2,0,1)).reshape(V, T*N), dout.reshape(N*T,D))

  # print dW.shape
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  N, D = x.shape
  N, H = prev_h.shape
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  temp = sigmoid(a[:, 0 : 3 * H])
  i, f, o = temp[:, 0 : H], temp[:, H : 2 * H], temp[:, 2 * H : 3 * H]
  g = np.tanh(a[:, 3 * H:4 * H])

  next_c = f*prev_c + i*g
  z = np.tanh(next_c)
  next_h = o*z

  cache = (i, f, o, g, z, prev_h, prev_c, x, Wx, Wh)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.

  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  (i, f, o, g, z, prev_h, prev_c, x, Wx, Wh) = cache
  N, H = prev_h.shape
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  do = dnext_h*z
  dWxo = np.dot(x.T, do*(1-o)*o)       #shape of (D,H)
  dWho = np.dot(prev_h.T, do*(1-o)*o)
  dbo  = np.sum(do*(1-o)*o, axis=0)
  dx = np.dot(do*(1-o)*o, Wx[:,2*H:3*H].T)
  dprev_h = np.dot(do*(1-o)*o, Wh[:,2*H:3*H].T)


  dz = dnext_h*o
  dct = dz*(1-z**2) + dnext_c

  di = dct*g
  dWxi = np.dot(x.T, di*(1-i)*i)
  dWhi = np.dot(prev_h.T, di*(1-i)*i)
  dbi = np.sum(di*(1-i)*i, axis=0)
  dx += np.dot(di*(1-i)*i, Wx[:,0:H].T)
  dprev_h += np.dot(di*(1-i)*i, Wh[:,0:H].T)

  dg = dct*i
  dWxg = np.dot(x.T, dg*(1-g**2))
  dWhg = np.dot(prev_h.T, dg*(1-g**2))
  dbg = np.sum(dg*(1-g**2), axis=0)
  dx += np.dot(dg*(1-g**2), Wx[:, 3*H:4*H].T)
  dprev_h += np.dot(dg*(1-g**2), Wh[:, 3*H:4*H].T)


  df = dct*prev_c
  dprev_c = dct*f

  dWxf = np.dot(x.T, df*(1-f)*f)
  dWhf = np.dot(prev_h.T, df*(1-f)*f)
  dbf = np.sum(df*(1-f)*f, axis=0)
  dx += np.dot(df*(1-f)*f, Wx[:, H:2*H].T)
  dprev_h += np.dot(df*(1-f)*f, Wh[:, H:2*H].T)

  dWx = np.concatenate((dWxi, dWxf, dWxo, dWxg), axis = 1)
  dWh = np.concatenate((dWhi, dWhf, dWho, dWhg), axis = 1)
  db  = np.concatenate((dbi, dbf, dbo, dbg), axis=0)


  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.

  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.

  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)

  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, {}
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  prev_c = np.zeros_like(h0)
  N, H = h0.shape
  h = np.zeros((N, T, H))
  h[:, 0, :], c, cache[0] = lstm_step_forward(x[:, 0, :], h0, prev_c, Wx, Wh, b)
  for t in range(T-1):
      h[:,t+1, :], c, cache[t+1] = lstm_step_forward(x[:,t+1,:], h[:, t, :], c, Wx, Wh, b)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]

  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  (N, T, H) = dh.shape
  (_, _, _, _, _, _, _, x, _, _) = cache[T-1]
  N, D = x.shape
  dnext_c = np.zeros((N, H))
  dx = np.zeros((N, T, D), dtype = np.float32)
  dWx = np.zeros((D, 4*H), dtype = np.float32)
  dWh = np.zeros((H, 4*H), dtype = np.float32)
  db = np.zeros((4*H), dtype = np.float32)

  dx[:,T-1,:], dnext_h, dnext_c, dWx, dWh, db = lstm_step_backward(dh[:, T-1, :], dnext_c,  cache[T-1])


  for t in range(T-2,-1,-1):
    grad_h = dnext_h + dh[:,t,:]
    dx[:,t,:], dnext_h, dnext_c, DWx, DWh, Db = lstm_step_backward(grad_h, dnext_c, cache[t])
    dWx = dWx + DWx
    dWh += DWh
    db  += Db
  # print dWx.shape, dWh.shape, db.shape
  dh0 = dnext_h

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)

  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape

  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)

  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]

  if verbose: print 'dx_flat: ', dx_flat.shape

  dx = dx_flat.reshape(N, T, V)

  return loss, dx
