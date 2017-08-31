import string
from collections import Counter


PUNCTUATION = set(string.punctuation)


def _simple_featurize_text(text):
    featuredict = Counter()
    for t in text.split():  # split on whitespace
        key = t.lower()  # lowercase
        key = ''.join([c for c in key if c not in PUNCTUATION])  # one way to remove punctuation (works in both Python 2 and 3)
        featuredict[key] += 1  # increment count
    return featuredict


def featurize_text(text):
    '''Bonus: extend featurize_text() to do more than just what's in
    _simple_featurize_text(). For example, you might start by excluding
    stopwords. A next step might be adding support for bigrams ("hot dog")
    instead of just unigrams ("hot", "dog"). What else can you think of?
    '''
    return _simple_featurize_text(text)
