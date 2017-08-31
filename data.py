import copy
import random
import json
import pprint
from features import featurize_text
from collections import namedtuple


DataPoint = namedtuple('DataPoint', ['raw_data', 'feature_dict', 'klass'])


def load_json_files(datasource_name_and_location, verbose=False):
    # Load data into memory (our data is small enough we can work with it in memory easily)
    scraped_pages = {}
    for name, filepath in datasource_name_and_location:
        with open(filepath) as json_data:
            scraped_pages[name] = json.load(json_data)

    if verbose:
        # View just one data point
        print("One post from NYT:")
        pprint.pprint([p['text'] for p in scraped_pages['newyorktimes'][110]['posts']])


def build_dataset(scraped_pages):
    # Build a featurized data set
    dataset = []
    for datasource_name, all_datasource_data in scraped_pages.items():
        for scraped_page in all_datasource_data:
            if 'posts' in scraped_page:
                for post in scraped_page['posts']:
                    if 'text' in post:
                        text = post['text']
                        text_features = featurize_text(text)
                        data_point = DataPoint(raw_data=text,
                                               feature_dict=text_features,
                                               klass=datasource_name)
                        dataset.append(data_point)

    return dataset


def split_dataset(dataset, fraction_train=0.8):
    dataset = copy.deepcopy(dataset)  # We copy our data here to avoid shuffling the original passed-in dataset list. This is expensive memory-wise (because we now have two copies of our dataset in memory rather than just one), but it keeps us from modifying the passed-in dataset outside this function, which is nice.  Overall there are better ways to structure code flow in terms of memory efficiency; this example is more for clarity of algorithm than efficiency of memory.
    random.shuffle(dataset)
    split_index = int(len(dataset) * fraction_train)
    train_dataset, test_dataset = dataset[:split_index], dataset[split_index:]
    return train_dataset, test_dataset
