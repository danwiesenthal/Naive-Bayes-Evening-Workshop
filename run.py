from data import load_json_files, build_dataset, split_dataset
from classifiers import NaiveBayesClassifier, evaluate_classifier


if __name__ == '__main__':
    print("Ok let's go!")

    # Where to find data
    datasource_info = [('newyorktimes', 'data/nyt_discussions.json'),
                       ('motherjones', 'data/motherjones_discussions.json'),
                       ('breitbart', 'data/breitbart_discussions.json')]

    # Load the dataset into memory
    json_text = load_json_files(datasource_info, verbose=True)
    dataset = build_dataset(json_text)

    # Split our data into train and test
    train_dataset, test_dataset = split_dataset(dataset, fraction_train=0.8)

    # Train our classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(train_dataset)

    # Evaluate our classifier, for each class
    performance_string = 'Class {} performance: f1={}, precision={}, recall={}'
    for klass in nb_classifier.class_counter:
        f1, precision, recall = evaluate_classifier(nb_classifier, klass,
                                                    test_dataset, verbose=False)

        print(performance_string.format(klass, f1, precision, recall))
