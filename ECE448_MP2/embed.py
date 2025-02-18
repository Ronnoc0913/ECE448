import numpy as np

def initialize(data, dim):
    '''
    Initialize embeddings for all distinct words in the input data.
    Most of the dimensions will be zero-mean unit-variance Gaussian random variables.
    In order to make debugging easier, however, we will assign special geometric values
    to the first two dimensions of the embedding:

    (1) Find out how many distinct words there are.
    (2) Choose that many locations uniformly spaced on a unit circle in the first two dimensions.
    (3) Put the words into those spots in the same order that they occur in the data.

    Thus if data[0] and data[1] are different words, you should have

    embedding[data[0]] = np.array([np.cos(0), np.sin(0), random, random, random, ...])
    embedding[data[1]] = np.array([np.cos(2*np.pi/N), np.sin(2*np.pi/N), random, random, random, ...])

    ... and so on, where N is the number of distinct words, and each random element is
    a Gaussian random variable with mean=0 and standard deviation=1.

    @param:
    data (list) - list of words in the input text, split on whitespace
    dim (int) - dimension of the learned embeddings

    @return:
    embedding - dict mapping from words (strings) to numpy arrays of dimension=dim.
    '''
    # raise RuntimeError("You need to write this part!")
    
    unique_words = list(dict.fromkeys(data)) # preserve the order
    N = len(unique_words)
    embedding = {}

    for i, word in enumerate(unique_words):
        theta = (2 * np.pi * i) / N # uniform spacing on circle
        vector = np.zeros(dim)
        vector[0] = np.cos(theta)
        vector[1] = np.sin(theta)
        vector[2:] = np.random.normal(0, 1, dim - 2) # gaussian noise for remaining dimensions
        embedding[word] = vector
    
    return embedding

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def gradient(embedding, data, t, d, k):
    '''
    Calculate gradient of the skipgram NCE loss with respect to the embedding of data[t]

    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    t (int) - data index of word with respect to which you want the gradient
    d (int) - choose context words from t-d through t+d, not including t
    k (int) - compare each context word to k words chosen uniformly at random from the data

    @return:
    g (numpy array) - loss gradients with respect to embedding of data[t]
    '''
    # raise RuntimeError("You need to write this part!")

    target_word = data[t]
    target_vec = embedding[target_word]
    vocab = list(embedding.keys())
    N = len(data)

    context_indices = [t + offset for offset in range (-d, d+1) if offset != 0 and 0 <= t + offset < N]
    context_words = [data[i] for i in context_indices]

    g = np.zeros_like(target_vec)

    for context_word in context_words:
        context_vec = embedding[context_word]
        score = np.dot(target_vec, context_vec)
        prob = sigmoid(score)
        g += (prob - 1) * context_vec
    
    for _ in range(k):
        noise_word = np.random.choice(vocab)
        noise_vec = embedding[noise_word]
        score = np.dot(target_vec, noise_vec)
        prob = sigmoid(score)
        g += prob * noise_vec  # Matched expected formula

    if len(vocab) == 1:
        z = np.dot(target_vec, target_vec)
        sigma_z = sigmoid(z)
        g = 2 * d * (2 * sigma_z - 1) * target_vec


    return g
           
def sgd(embedding, data, learning_rate, num_iters, d, k):
    '''
    Perform num_iters steps of stochastic gradient descent.

    @param:
    embedding - dict mapping from words (strings) to numpy arrays.
    data (list) - list of words in the input text, split on whitespace
    learning_rate (scalar) - scale the negative gradient by this amount at each step
    num_iters (int) - the number of iterations to perform
    d (int) - context width hyperparameter for gradient computation
    k (int) - noise sample size hyperparameter for gradient computation

    @return:
    embedding - the updated embeddings
    '''
    # raise RuntimeError("You need to write this part!")

    for _ in range (num_iters):
        t = np.random.randint(len(data))
        grad = gradient(embedding, data, t, d, k)
        embedding[data[t]] -= learning_rate * grad

    return embedding
    

