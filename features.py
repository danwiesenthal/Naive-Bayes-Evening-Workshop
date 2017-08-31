from string import punctuation
from collections import Counter


PUNCTUATION = set(punctuation)


def _simple_featurize_text(text):
    feature_dict = Counter()
    for key in text.lower().split(): # lowercase and split on whitespace
        key = ''.join([c for c in key if c not in PUNCTUATION]) # remove punctuation
        feature_dict[key] += 1 # increment count
    return feature_dict


def featurize_text(text):
    '''Bonus: extend featurize_text() to do more than just what's in
    _simple_featurize_text(). For example, you might start by excluding
    stopwords. A next step might be adding support for bigrams ("hot dog")
    instead of just unigrams ("hot", "dog"). What else can you think of?
    '''
    return _simple_featurize_text(text)
