"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import defaultdict, Counter
import math

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    word_tag_counts = defaultdict(Counter)
    tag_counts = Counter()
    
    # Count occurrences of words with each tag
    for sentence in train:
        for word, tag in sentence:
            word_tag_counts[word][tag] += 1
            tag_counts[tag] += 1
    
    # Determine most frequent tag for each word
    most_common_tag = tag_counts.most_common(1)[0][0]  # Most frequent tag overall
    word_most_common_tag = {word: max(tags, key=tags.get) for word, tags in word_tag_counts.items()}
    
    # Predict tags
    predictions = []
    for sentence in test:
        tagged_sentence = [(word, word_most_common_tag.get(word, most_common_tag)) for word in sentence]
        predictions.append(tagged_sentence)
    
    return predictions