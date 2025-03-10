import sys, random
import numpy as np
import reader

'''
Perform one layer of transformer inference on a dataset
using embeddings, positional_embeddings, and weight matrices 
specified in the file model.json
'''

def softmax(logits):
    '''
    Return the row-wise softmax of a matrix.  
    @param:
    logits - any numpy array
    @return:
    probs - probs[i,j] = exp(logits[i,j])/sum(exp(logits[i,:])), but 
      be careful to normalize so that you avoid overflow errors!
    '''
    raise RuntimeError("You need to write this part!")

def forward(XK, XQ, WK, WO, WQ, WV):
    '''
    Perform one layer of transformer inference, using trained model, on given data.

    @param:
    XK - (T-2)-by-V array containing embeddings of words to be used for keys and values
    XQ - 2-by-V array containing embeddings of words to be used for queries
    WK - V-by-d array mapping X to K
    WO - d-by-V array mapping C to O
    WQ - V-by-d array mapping X to Q
    WV - V-by-d array mapping X to V

    @return:
    A - 2-by-(T-2) array, A[i,j] is attention the i'th query pays to the j'th key
    C - 2-by-d array, context vectors from which P is computed
    K - (T-2)-by-d array, key vectors computed from XK
    O - 2-by-V array, O[i,j] is probability that i'th output word should be j
    Q - 2-by-d array, query vectors computed from XQ
    V - (T-2)-by-d array, value vectors computed from XK
    '''
    raise RuntimeError("You need to write this part!")
    return A, C, K, O, Q, V


def generate(embeddings, vocabulary, WK, WO, WQ, WV):
    '''
    Perform inference on the provided embeddings, and report the generated sentences.
    
    @param:
    embeddings - a list of one-hot embedding matrices, one per sentence
    vocabulary - a list of words in the vocabulary
    WK - V-by-d array mapping X to K
    WO - d-by-V array mapping C to O
    WQ - V-by-d array mapping X to Q
    WV - V-by-d array mapping X to V

    @return:
    generated - a list of generated sentences, each as a list of space-separated words.
      The first T-2 words of each sentence should be vocabulary items indexed by the
      argmax of the first T-2 embeddings.  The last 2 words of each sentence should be
      vocabulary items indexed by the argmax of the two outputs computed by running
      the transformer with the provided WK, WO, WQ, and WV.
    '''
    raise RuntimeError("You need to write this part!")
    return generated

def cross_entropy_loss(O, Y):
    '''
    Calculate losses from network outputs O if target one-hot vectors are Y.

    @param:
    O - NQ-by-V array.  O[n,v]=probability that n'th output is v.
    Y - NQ-by-V array. Y[n,v]=1 if n'th target is v, else Y[n,v]=0.
    
    @return:
    L - cross-entropy loss, summed over all rows
    dO - NQ-by-V array.  Derivatives of the loss with respect to the elements of O.
    '''
    raise RuntimeError("You need to write this part!")
    return L, dO

def gradient(XK, XQ, Y, WK, WO, WQ, WV, A, C, K, O, Q, V):
    '''
    Compute gradient of cross-entropy loss with respect to WK, WO, WQ, and WV
    given the input data in K, Q, and V, and the target outputs in Y.
    
    @param:
    XK - one embedding per row, first n-2 words in the sentence
    XQ - one embedding per row, 3rd-from-last and 2nd-from-last words in the sentence
    Y - one embedding per row, last two words in the sentence
    O - 2-by-V array, O[i,j] is probability that i'th output word should be j
    C - 2-by-d array, context vectors from which O is computed
    V - (T-2)-by-d array, value vectors of which each row of C is a weighted average
    A - 2-by-(T-2) array, A[i,j] is attention the i'th query pays to the j'th key
    K - (T-2)-by-d array, key vectors computed from XK
    Q - 2-by-d array, query vectors computed from XQ

    @return:
    dWK - gradient of cross-entropy with respect to WK
    dWO - gradient of cross-entropy with respect to WO
    dWQ - gradient of cross-entropy with respect to WQ
    dWV - gradient of cross-entropy with respect to WV
    '''
    raise RuntimeError("You need to write this part!")
    return dWK, dWO, dWQ, dWV
    

def train(embeddings, WK, WO, WQ, WV, learningrate, num_iters):
    '''
    Train a transformer using stochastic gradient descent (SGD).
    Each iteration of SGD should choose one training sentence, uniformly at random,
    compute the loss and loss gradient for that one sentence,
    then adjust the parameters WK, WO, WQ and WV in the direction of the negative
    gradient scaled by the learningrate.

    @param:
    embeddings - embeddings[i][j,:] is one-hot vector of the j'th word in the i'th training sentence
    WK - the matrix that multiplies each embedding to produce a key
    WO - the matrix that multiplies the context vector to produce an output logit vector
    WQ - the matrix that multiplies each embedding to produce a query
    WV - the matrix that multiplies each embedding to produce a value
    learningrate - scalar learning rate
    num_iters - number of iterations of SGD to perform

    @return:
    losses - losses[t]=cross-entropy loss of t'th iteration
    WK - what WK has become after num_iters of training
    WO - what WO has become after num_iters of training
    WQ - what WQ has become after num_iters of training
    WV - what WV has become after num_iters of training
    '''
    raise RuntimeError("You need to write this part!")
    return losses, WK, WO, WQ, WV

