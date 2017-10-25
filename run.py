from data import split_dataset
from features import featurize_text
from classifiers import NaiveBayesClassifier, evaluate_classifier
from datasource import load_json_files, build_dataset


if __name__ == '__main__':
    print("Ok let's go!")

    # Where to find data
    datasource_info = [('newyorktimes', 'data/nyt_discussions.json'),
                       ('motherjones', 'data/motherjones_discussions.json'),
                       ('breitbart', 'data/breitbart_discussions.json')]

    # Load the dataset into memory
    json_text = load_json_files(datasource_info, verbose=True)
    dataset = build_dataset(json_text, featurize_text, verbose=True)

    # Split our data into train and test
    train_dataset, test_dataset = split_dataset(dataset, fraction_train=0.8)

    # Train our classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(train_dataset)

    # Evaluate our classifier, for each class
    performance_string = 'Class {klass} performance: f1={f1:.{digits}}, precision={precision:.{digits}}, recall={recall:.{digits}}'
    for klass in sorted(nb_classifier.class_counter):  # sort just for nicer output
        f1, precision, recall = evaluate_classifier(nb_classifier, klass, test_dataset)
        print(performance_string.format(klass=klass, f1=f1, precision=precision, recall=recall, digits=3))
