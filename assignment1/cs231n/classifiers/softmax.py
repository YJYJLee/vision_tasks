import numpy as np
from random import shuffle
from past.builtins import xrange

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

  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores=np.dot(X[i],W)
    scores -= np.max(scores)
    
    correct_class_score=scores[y[i]]
    loss -= np.log(np.exp(correct_class_score)/np.sum(np.exp(scores)))
    #print(np.exp(scores))
    
       
    for j in xrange(num_classes):
        exp = np.exp(scores[j])/np.sum(np.exp(scores))
        if j==y[i]:
            dW[:,j] += (-1+exp) * X[i]
        else:
            dW[:,j] += exp * X[i]
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train               #average loss
  dW /= num_train                 #average dW

  # Add regularization to the loss.
  loss +=  reg * np.sum(W * W)
  dW += 2 * reg * W

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
  
  num_classes = W.shape[1]
  num_train = X.shape[0]
    

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores= X.dot(W)       # scores = [N,C]
  scores -= np.max(scores, axis = 1).reshape(-1,1)    # [N, C]
#  print(scores.shape[0],scores.shape[1])
#  print(scores[0])
#  print(scores[1])
#  print(np.max(scores[range(num_train)],axis=1)) 
#  score_split=np.split(np.max(scores[range(num_train)],axis=1),num_train)
#  print(np.max(scores[range(num_train)],axis=1).shape[1])
#score_split = np.tile((np.max(scores[range(num_train)],axis=1),num_classes)
#print(score_split.shape[0])
                        
  #scores -= np.tile(score_split,(num_train,num_classes))
  #print(scores.shape[0],scores.shape[1])
#  correct_class_score = scores[range(num_train),list(y)].reshape(-1,1)

  
  exp = np.exp(scores)/np.sum(np.exp(scores),axis=1).reshape(-1,1)
  #print(exp.shape[0],exp.shape[1])
  loss = -np.sum(np.log(exp[range(num_train),list(y)]))
  loss /= num_train   
  loss +=  reg * np.sum(W * W)

  dW = exp.copy()
  dW[range(num_train),list(y)] -=1
  dW = (X.T).dot(dW)
    
  dW /= num_train  
  dW += 2 * reg * W
  #print(num_train, num_classes)
  #dW[range(num_train),list(y)] += (-1+exp) * X[range(num_train)]
  '''for j in xrange(num_classes):
        exp = np.exp(scores[j])/np.sum(np.exp(scores))
        if j==y[i]:
            dW[:,j] += (-1+exp) * X[i]
        else:
            dW[:,j] += exp * X[i]
  '''  

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

