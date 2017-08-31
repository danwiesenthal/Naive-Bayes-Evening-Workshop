from collections import Counter, defaultdict
import math
import pprint
import functools


#  Advanced material here, feel free to ignore memoize if you like.  Long story short, it remembers the inputs and outputs to functions, and if the same input is seen multiple times, then rather than running the function multiple times, memoization just returns the result of the function when it was first called with that input.  Memory expensive because it keeps track of past results, but computationally nice because we don't recalculate the same thing over and over.
def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer


def evaluate_classifier(classifier, class_of_interest,
                        evaluation_data, verbose=False, progress=True):
    if verbose:
        print("Evaluating performance for class {}".format(class_of_interest))
    tp, fp, tn, fn = 0, 0, 0, 0  # true positive, false positive, true negative, false negative
    count = 0
    for dp in evaluation_data:
        count += 1
        if progress:
            if count % 1000 == 0:
                print("progress: {} / {}".format(count, len(evaluation_data)))
        prediction = classifier.predict(dp)
        actual = dp.klass
        if actual == prediction:  # we got it right!
            if prediction == class_of_interest:
                tp += 1
            else:
                tn += 1
        else:  # we got it wrong :(
            if prediction == class_of_interest:
                fp += 1
            else:
                fn += 1
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    if verbose:
        print("precision:", precision)
        print("recall:", recall)
        print("f1:", f1)
    return f1, precision, recall


class NaiveBayesClassifier(object):

    def __init__(self, laplace_smoothing_constant=0.01):
        self.total_counter = 0
        self.class_counter = Counter()
        self.feature_given_class_counter = defaultdict(Counter)

        # Hyperparameter that can be tuned via Cross Validation to improve performance
        self.laplace_smoothing_constant = laplace_smoothing_constant

    def _update_with_one_data_point(self, data_point):
        # Increment the total counter
        self.total_counter += 1

        # Increment class_counter
        self.class_counter[data_point.klass] += 1

        # Increment feature_given_class counter for each feature in featuredict
        for feature_name, feature_value in data_point.featuredict.items():
            assert type(feature_value) == int, "only int typed feature values currently supported"
            # Bonus: can one extend Naive Bayes to real-valued features?  (hint: yes)
            self.feature_given_class_counter[data_point.klass][feature_name] += feature_value

    def train(self, train_set, verbose=False):
        for data_point in train_set:
            self._update_with_one_data_point(data_point)
        if verbose:
            print("Training complete. Counters:")
            pprint.pprint(self.total_counter)
            pprint.pprint(self.class_counter)
            pprint.pprint(self.feature_given_class_counter)

    @memoize  # Advanced material, see note on memoize above
    def _prior(self, klass):
        # Laplace smoothing
        numerator = self.laplace_smoothing_constant
        denominator = len(self.class_counter) * self.laplace_smoothing_constant

        # On top of the unsmoothed counts
        numerator += self.class_counter[klass]
        denominator += self.total_counter

        # Gives us our smoothed prior
        return float(numerator) / denominator

    @memoize  # Advanced material, see note on memoize above
    def _vocabulary_size(self):
        vocab = set()
        for klass in self.class_counter:  # for each class
            # get all the features in class and add them to total cross-class vocabulary
            vocab.update(set(self.feature_given_class_counter[klass]))
        return len(vocab)

    @memoize  # Advanced material, see note on memoize above
    def _likelihood(self, klass, feature_name):
        # Laplace smoothing
        numerator = self.laplace_smoothing_constant
        denominator = self._vocabulary_size() * self.laplace_smoothing_constant

        # On top of the unsmoothed counts
        numerator += self.feature_given_class_counter[klass].get(feature_name, 0)
        denominator += sum(self.feature_given_class_counter[klass].values())

        # Gives us our smoothed likelihood
        return float(numerator) / denominator

    def predict(self, data_point, verbose=False):
        # Where we'll store probabilities by class
        pseudo_probability_by_class = {}

        # Calculate the pseudo probability for each class
        for klass in self.class_counter:
            prior = self._prior(klass)

            # Aggregate likelihood
            likelihoods = []
            for feature_name in data_point.featuredict:  # for each feature
                # for each time the feature appeared
                for _ in range(data_point.featuredict[feature_name]):
                    likelihoods.append(self._likelihood(klass, feature_name))

            # Add prior and likelihoods in logspace to avoid floating point underflow.
            # The class with the highest log probability is still the most probable.
            numerator_terms = [prior] + likelihoods
            pseudo_probability_by_class[klass] = sum([math.log(t) for t in numerator_terms])

        # Pick the class with the maximum probability and return it as our prediction
        sorted_probability_by_class = sorted(pseudo_probability_by_class.items(),
                                             # Sorts ascending by default, we want biggest probability first => descending
                                             key=lambda x: x[1], reverse=True)
        return sorted_probability_by_class[0][0]
