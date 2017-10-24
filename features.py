import string
from collections import Counter


PUNCTUATION = set(string.punctuation)


def _simple_featurize_text(text):
    '''Bonus (beginner): extend _simple_featurize_text() to remove common words, known as stopwords, like "the" and "and" since it's generally found that they don't add much signal. What other words might make sense to remove (because they don't add signal) in the context of news articles or online comments?  (Hint: try thinking both about words that occur so commonly that they have little signal, as well as words that occur super rarrrrrely and therefore might nottt havve siggnal ;) '''
    featuredict = Counter()
    for t in text.split():  # split on whitespace
        key = t.lower()  # lowercase
        key = ''.join([c for c in key if c not in PUNCTUATION])  # one way to remove punctuation (works in both Python 2 and 3)
        featuredict[key] += 1  # increment count
    return featuredict


def featurize_text(text):
    '''Bonus (intermediate): extend featurize_text() to do more than just what's in
    _simple_featurize_text(). For example, you might start by excluding
    stopwords. A next step might be adding support for bigrams ("hot dog")
    instead of just unigrams ("hot", "dog"). What else can you think of?
    '''
    return _simple_featurize_text(text)
