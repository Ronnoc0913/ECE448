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
    is_1d = False
    if len(logits.shape) == 1:
        logits = logits[np.newaxis, :]
        is_1d = True

    row_maxes = np.max(logits, axis=1, keepdims=True)  # shape: (N, 1)
    shifted = logits - row_maxes
    exp_vals = np.exp(shifted)
    sums = np.sum(exp_vals, axis=1, keepdims=True)     # shape: (N, 1)
    probs = exp_vals / sums                            # shape: (N, V)

    # If input was 1D, squeeze back to 1D
    if is_1d:
        return probs[0]
    else:
        return probs

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
     # 1) Compute K, Q, V by multiplying embeddings by the parameter matrices
    K = XK @ WK  # (T-2, d)
    Q = XQ @ WQ  # (2, d)
    Vv = XK @ WV # (T-2, d)

    # 2) Compute attention logits = QK^T -> shape = (2, T-2)
    attn_logits = Q @ K.T

    # 3) Softmax row-wise for attention weights
    A = softmax(attn_logits)  # shape: (2, T-2)

    # 4) Compute context vectors: C = A @ V
    #    A: (2, T-2), Vv: (T-2, d) => C: (2, d)
    C = A @ Vv

    # 5) Compute output logits = C @ WO -> shape (2, V)
    logits = C @ WO

    # 6) Output probabilities = softmax(logits) row-wise
    O = softmax(logits)

    return A, C, K, O, Q, Vv


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
    generated = []

    for emb in embeddings:
        # Split this sentence into XK, XQ, Y via define_task
        XK, XQ, Y = reader.define_task(emb)   # XK:(T-2)xV, XQ:2xV, Y:2xV

        # The first T-2 words are the 'prompt' words in XK
        # We'll retrieve them by argmax of each row
        prompt_ids = np.argmax(XK, axis=1)
        prompt_words = [vocabulary[i] for i in prompt_ids]

        # Forward pass to get final 2 outputs
        A, C, K, O, Q, Vv = forward(XK, XQ, WK, WO, WQ, WV)

        # For each row in O, pick the word with highest probability
        # O has shape 2 x V
        out_ids = np.argmax(O, axis=1)  # 2-element array
        out_words = [vocabulary[i] for i in out_ids]

        # Combine the prompt words and the newly generated last 2 words
        full_sentence = prompt_words + out_words
        generated.append(full_sentence)

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

    eps = sys.float_info.min  # small constant to avoid log(0)
    # O might have zeros, so clip or max with eps to avoid log(0)
    O_clipped = np.maximum(O, eps)

    # Cross-entropy
    L = -np.sum(Y * np.log(O_clipped))

    # Derivative w.r.t. O
    dO = - (Y / O_clipped)  # shape same as O

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
    # 1) Cross-entropy derivative wrt O
    L, dO = cross_entropy_loss(O, Y)  # shape 2 x V

    # 2) Derivative w.r.t. 'logits' (the input to softmax).
    #    O = softmax(logits). For each row i:
    #      dL/dlogits[i,v] = O[i,v]*(dO[i,v] - sum_over_x( O[i,x]*dO[i,x] ))
    NQ, Vdim = O.shape  # typically (2, V)
    dlogits = np.zeros_like(O)
    for i in range(NQ):
        # sum of O[i,:]*dO[i,:]
        row_s = np.sum(O[i] * dO[i])
        for v_ in range(Vdim):
            dlogits[i,v_] = O[i,v_] * (dO[i,v_] - row_s * O[i,v_])

    # 3) From logits = C @ WO => shape (2, V)
    #    dC = dlogits @ WO^T
    #    dWO = C^T @ dlogits
    dC = dlogits @ WO.T  # shape (2, d)
    dWO = C.T @ dlogits  # shape (d, V)

    # 4) From C = A @ V => shape (2, d)
    #    dA = dC @ V^T
    #    dV = A^T @ dC
    dA = dC @ V.T          # shape (2, T-2)
    dVv = A.T @ dC         # shape (T-2, d)

    # 5) A = softmax(QK^T). Let Z = QK^T => shape (2, T-2)
    #    dZ is row-by-row the derivative of softmax
    dZ = np.zeros_like(A)  # shape (2, T-2)
    for i in range(A.shape[0]):  # 2
        # sum of A[i,:]*dA[i,:]
        row_s = np.sum(A[i] * dA[i])
        for j in range(A.shape[1]):  # T-2
            dZ[i,j] = A[i,j] * (dA[i,j] - row_s * A[i,j])

    # 6) Z = QK^T => shape (2, T-2)
    #    dQ = dZ @ K
    #    dK = dZ^T @ Q
    dQ = dZ @ K            # shape (2, d)
    dK = dZ.T @ Q          # shape (T-2, d)

    # 7) K = XK @ WK => shape (T-2, d)
    #    dWK = XK^T @ dK
    dWK = XK.T @ dK        # shape (V, d)

    # 8) Q = XQ @ WQ => shape (2, d)
    #    dWQ = XQ^T @ dQ
    dWQ = XQ.T @ dQ        # shape (V, d)

    # 9) V = XK @ WV => shape (T-2, d)
    #    dWV = XK^T @ dVv
    dWV = XK.T @ dVv       # shape (V, d)

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
    import random

    losses = []
    N = len(embeddings)  # number of training sentences

    for t in range(num_iters):
        # pick random sample
        i = random.randint(0, N-1)
        emb = embeddings[i]

        # define task
        XK, XQ, Y = reader.define_task(emb)

        # forward pass
        A, C, K, O, Q, Vv = forward(XK, XQ, WK, WO, WQ, WV)

        # compute gradient
        dWK, dWO, dWQ, dWV = gradient(XK, XQ, Y, WK, WO, WQ, WV, A, C, K, O, Q, Vv)

        # cross-entropy loss
        L, _ = cross_entropy_loss(O, Y)
        losses.append(L)

        # parameter update
        WK -= learningrate * dWK
        WO -= learningrate * dWO
        WQ -= learningrate * dWQ
        WV -= learningrate * dWV

        # (optional) print progress every so often
        # if t % 1000 == 0:
        #     print(f"Iteration {t}, loss={L:.4f}")

    return losses, WK, WO, WQ, WV
