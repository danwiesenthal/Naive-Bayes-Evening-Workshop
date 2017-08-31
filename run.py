from data import load_scraped_json_files_into_DataPoint_objects, split_dataset
from features import featurize_text
from classifiers import NaiveBayesClassifier, evaluate_classifier


if __name__ == '__main__':
    print("Ok let's go!")

    # Where to find data
    datasource_name_and_location = [('newyorktimes', 'data/nyt_discussions.json'), ('motherjones', 'data/motherjones_discussions.json'), ('breitbart', 'data/breitbart_discussions.json')]

    # Load the dataset into memory
    dataset = load_scraped_json_files_into_DataPoint_objects(datasource_name_and_location, verbose=True)

    # Featurize our data
    for data_point in dataset:
        data_point.featuredict = featurize_text(data_point.raw_data)

    # Split our data into train and test
    train_dataset, test_dataset = split_dataset(dataset, fraction_train=0.8)

    # Train our classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(train_dataset)

    # Evaluate our classifier, for each class
    for klass in nb_classifier.class_counter:
        f1, precision, recall = evaluate_classifier(nb_classifier, klass, test_dataset, verbose=False)
        print("Performance for class {}:  f1={},  precision={},  recall={}".format(klass, f1, precision, recall))
