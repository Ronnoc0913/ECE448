"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.

    tag_counts = Counter()
    trans_counts = defaultdict(Counter)
    emit_counts = defaultdict(Counter)
    unique_tags = set()
    vocab = set()

    for sentence in sentences:
        prev_tag = "START"
        for word, tag in sentence:
            tag_counts[tag] += 1
            emit_counts[tag][word] += 1
            trans_counts[prev_tag][tag] += 1
            unique_tags.add(tag)
            vocab.add(word)
            prev_tag = tag
        trans_counts[prev_tag]["END"] += 1

    V_trans = len(unique_tags) + 1 
    for prev_tag in trans_counts:
        total = sum(trans_counts[prev_tag].values())
        for curr_tag in unique_tags | {"END"}:  
            count = trans_counts[prev_tag].get(curr_tag, 0)
            trans_prob[prev_tag][curr_tag] = (count + emit_epsilon) / (total + emit_epsilon * V_trans)

    for tag in emit_counts:
        V_emiss = len(emit_counts[tag])  
        total_emiss = tag_counts[tag]
        denom = total_emiss + emit_epsilon * (V_emiss + 1)

        for word in emit_counts[tag]:
            emit_prob[tag][word] = (emit_counts[tag][word] + emit_epsilon) / denom
        
        emit_prob[tag]["UNKNOWN"] = emit_epsilon / denom

    for tag in unique_tags:
        init_prob[tag] = trans_prob["START"].get(tag, 0.0)

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.
    log_prob = {} 
    predict_tag_seq = {} 
    
    for curr_tag in emit_prob:
        max_prob, best_prev_tag = max(
            ((prev_prob[prev_tag] + log(trans_prob.get(prev_tag, {}).get(curr_tag, epsilon_for_pt)), prev_tag)
             for prev_tag in prev_prob),
            key=lambda x: x[0]
        )
        
        observed_word_prob = emit_prob.get(curr_tag, {}).get(word, emit_prob[curr_tag].get("UNKNOWN", emit_epsilon))
        log_prob[curr_tag] = max_prob + log(observed_word_prob)
        predict_tag_seq[curr_tag] = prev_predict_tag_seq[best_prev_tag] + [curr_tag]
    
    return log_prob, predict_tag_seq



def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        best_final_tag = max(log_prob, key=log_prob.get)
        predicts.append(list(zip(sentence, predict_tag_seq[best_final_tag])))

    return predicts
