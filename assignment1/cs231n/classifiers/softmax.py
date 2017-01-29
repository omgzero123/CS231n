import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  #f = np.zeros((num_class,1))

  for i in xrange(num_train):
    #compute each score of X[i]
    f = X[i].dot(W)
    #shift f
    f -= np.max(f)
    #get probability
    p = np.exp(f) / np.sum(np.exp(f))
    #add to the loss 
    loss += -np.log(p[y[i]])
    
    #compute the dW
    
    for j in xrange(num_class):
      if j==y[i]:
        dW[:,j] += p[y[i]] * (np.sum(np.exp(f))-np.exp(f[y[i]])) * (-1) / np.exp(f[y[i]]) * X[i] 
      else:
        dW[:,j] += p[y[i]] * np.exp(f[j]) / np.exp(f[y[i]]) * X[i]
     
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  f = X.dot(W)
  maxf = np.max(f,axis=1)
  f -= np.tile(maxf,(num_class,1)).T

  sumexpf = np.sum(np.exp(f),axis=1)
  p = np.exp(f) / np.tile(sumexpf,(num_class,1)).T

  loss = np.sum(-np.log(p[range(0,num_train),y[0:num_train]]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  #dW = (p ./ e_f_yi .* e_f_j) * X / num_train + reg * W 
  efj = np.exp(f) 
  efj[range(0,num_train),y[0:num_train]] = (np.sum(efj,axis=1) - \
    np.exp(f[range(0,num_train),y[0:num_train]])) * (-1)
  dW = X.T.dot(np.tile(p[range(0,num_train),y[0:num_train]],(num_class,1)).T / \
    (np.tile(np.exp(f[range(0,num_train),y[0:num_train]]),(num_class,1)).T) * efj)

  dW /= num_train
  dW += reg * W 

  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

