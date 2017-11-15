import unittest
from interactive import PartialNaiveBayesClassifier
from data import DataPoint
from features import featurize_text
from collections import Counter, defaultdict
import interactive_solutions
from pprint import pprint


class TestInteractive(unittest.TestCase):

    example_dataset = [
        ["I love llamas", 'animals'],
        ["I love aligators", 'animals'],
        ["I love Ethiopian Cuisine", 'food'],
        ["I love Japanese Cuisine", 'food'],
        ["I love love itself", 'humanity']
    ]

    def setUp(self):
        self.nb = PartialNaiveBayesClassifier()
        self.nb_solution = interactive_solutions.PartialNaiveBayesClassifier()
        self.test_data_points = [DataPoint(d[0], featurize_text, klass=d[1]) for d in self.example_dataset]

    def test_sanity(self):
        self.assertEqual(1 + 1, 2, "One plus one should equal two")

    def test_initialize_total_counter(self):
        with self.assertRaises(AttributeError):
            self.nb.total_counter
        self.nb.initialize_total_counter()
        self.assertIsNotNone(self.nb.total_counter, "PartialNaiveBayesClassifier total_counter should NOT be None after being initialized")

    def test_initialize_class_counter(self):
        with self.assertRaises(AttributeError):
            self.nb.class_counter
        self.nb.initialize_class_counter()
        self.assertIsNotNone(self.nb.class_counter, "PartialNaiveBayesClassifier class_counter should NOT be None after being initialized")

    def test_initialize_feature_given_class_counter(self):
        with self.assertRaises(AttributeError):
            self.nb.feature_given_class_counter
        self.nb.initialize_feature_given_class_counter()
        self.assertIsNotNone(self.nb.feature_given_class_counter, "PartialNaiveBayesClassifier feature_given_class_counter should NOT be None after being initialized")

    def test_update_total_counter(self):
        self.nb.initialize_total_counter()
        for data_point in self.test_data_points:
            self.nb.update_total_counter(data_point)
        self.assertEqual(self.nb.total_counter, 5, "After training on 5 test data points, the total_counter should reflect the total number of data points upon which the classifier was trained")

    def test_update_class_counter(self):
        self.nb.initialize_class_counter()
        for data_point in self.test_data_points:
            self.nb.update_class_counter(data_point)
        self.assertEqual(self.nb.class_counter, Counter({'animals': 2, 'food': 2, 'humanity': 1}), "After training on 5 test data points, the class_counter should be a map of each class to the number of times it was seen in the data points upon which the classifier was trained")

    def test_update_feature_given_class_counter(self):
        self.nb.initialize_feature_given_class_counter()
        self.nb_solution.initialize_feature_given_class_counter()
        for data_point in self.test_data_points:
            self.nb.update_feature_given_class_counter(data_point)
            self.nb_solution.update_feature_given_class_counter(data_point)
        self.assertEqual(self.nb.feature_given_class_counter, self.nb_solution.feature_given_class_counter, "After training on 5 test data points, the feature_given_class_counter should offer a breakdown, per class, of how many times the feature occurred in that class")

    def test_prior(self):
        self.nb.initialize_total_counter()
        self.nb.initialize_class_counter()
        for data_point in self.test_data_points:
            self.nb.update_total_counter(data_point)
            self.nb.update_class_counter(data_point)
        self.assertEqual(self.nb.prior('animals'), 0.4, "After training on 5 test data points, what is the prior probability (proportion) of them which are about animals?")
        self.assertEqual(self.nb.prior('food'), 0.4, "After training on 5 test data points, what is the prior probability (proportion) of them which are about food?")
        self.assertEqual(self.nb.prior('humanity'), 0.2, "After training on 5 test data points, what is the prior probability (proportion) of them which are about humanity?")

    def test_likelihood(self):
        self.nb.initialize_class_counter()
        self.nb.initialize_feature_given_class_counter()
        for data_point in self.test_data_points:
            self.nb.update_class_counter(data_point)
            self.nb.update_feature_given_class_counter(data_point)
        self.assertEqual(self.nb.likelihood('llamas', 'animals'), 1.0 / 6, "After training on 5 test data points, what is the likelihood (probability of feature given class) of the feature 'llamas' being present when the class is 'animals'?")
        self.assertEqual(self.nb.likelihood('cuisine', 'food'), 2.0 / 8, "After training on 5 test data points, what is the likelihood (probability of feature given class) of the feature 'cuisine' being present when the class is 'food'?")
        self.assertEqual(self.nb.likelihood('love', 'humanity'), 2.0 / 4, "After training on 5 test data points, what is the likelihood (probability of feature given class) of the feature 'love' being present when the class is 'humanity'?")
        self.assertEqual(self.nb.likelihood('aligators', 'humanity'), 0.01 / 4, "After training on 5 test data points, what is the likelihood (probability of feature given class) of the feature 'aligators' being present when the class is 'humanity'?")


if __name__ == '__main__':
    print("\n\nUsing example dataset for testing:")
    pprint(TestInteractive.example_dataset)
    unittest.main()
