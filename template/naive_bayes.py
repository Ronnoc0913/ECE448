# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_labels, train_data, dev_data, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)

    # Count words in positive and negative reviews
    word_count_pos = Counter()
    word_count_neg = Counter()

    total_words_pos = 0
    total_words_neg = 0

    for i in range (len(train_data)):
        if train_labels[i] == 1:
            word_count_pos.update(train_data[i])
            total_words_pos += len(train_data[i])
        else:
            word_count_neg.update(train_data[i])
            total_words_neg += len(train_data[i])

    # Create vocabulary and compute probabilities in advance.
    vocabulary = set(word_count_pos.keys()).union(set(word_count_neg.keys()))
    vocab_size = len(vocabulary)

    p_word_given_pos = {
        word: math.log((word_count_pos[word] + laplace) / (total_words_pos + laplace * vocab_size))
        for word in vocabulary
    }

    p_word_given_neg = {
        word: math.log((word_count_neg[word] + laplace) / (total_words_neg + laplace * vocab_size))
        for word in vocabulary
    }

    # Handling unseen words
    unseen_prob_pos = math.log(laplace / (total_words_pos + laplace * vocab_size))
    unseen_prob_neg = math.log(laplace / (total_words_neg + laplace * vocab_size))

    # Compute Prior Probabilities (in log space)
    log_prior_pos = math.log(pos_prior)
    log_prior_neg = math.log(1 - pos_prior)

    # Classify Development Data
    yhats = []
    for review in tqdm(dev_data, disable=silently):
        log_prob_pos = log_prior_pos
        log_prob_neg = log_prior_neg

        for word in review:
            log_prob_pos += p_word_given_pos.get(word, unseen_prob_pos)
            log_prob_neg += p_word_given_neg.get(word, unseen_prob_neg)

        # Choose the label with the higher probability
        predicted_label = 1 if log_prob_pos > log_prob_neg else 0
        yhats.append(predicted_label)

    return yhats 

