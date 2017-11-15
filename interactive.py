# You'll find here some interactive functions that you can fill out during our workshop!


class PartialNaiveBayesClassifier(object):

    #  Training
    def initialize_total_counter(self):
        pass

    def update_total_counter(self, data_point):
        pass

    def initialize_class_counter(self):
        pass

    def update_class_counter(self, data_point):
        pass

    def initialize_feature_given_class_counter(self):
        pass

    def update_feature_given_class_counter(self, data_point):
        pass

    #  Prediction
    def prior(self, klass):
        pass

    def likelihood(self, feature_name, klass):
        pass
