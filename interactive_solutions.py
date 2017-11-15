# You'll find here some solutions to interactive functions that you can fill out during our workshop!

from collections import Counter, defaultdict


class PartialNaiveBayesClassifier(object):

    #  Training
    def initialize_total_counter(self):
        self.total_counter = 0

    def update_total_counter(self, data_point):
        self.total_counter += 1

    def initialize_class_counter(self):
        self.class_counter = Counter()

    def update_class_counter(self, data_point):
        self.class_counter[data_point.klass] += 1

    def initialize_feature_given_class_counter(self):
        self.feature_given_class_counter = defaultdict(Counter)

    def update_feature_given_class_counter(self, data_point):
        for feature_name, feature_value in data_point.featuredict.items():
            self.feature_given_class_counter[data_point.klass][feature_name] += feature_value

    #  Prediction
    def prior(self, klass):
        numerator = self.class_counter.get(klass, 0.01)  # See notes on Laplace Smoothing
        denominator = self.total_counter
        return float(numerator) / denominator

    def likelihood(self, feature_name, klass):
        numerator = self.feature_given_class_counter[klass].get(feature_name, 0.01)  # See notes on Laplace Smoothing
        denominator = sum(self.feature_given_class_counter[klass].values())
        return float(numerator) / denominator
