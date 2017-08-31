from collections import Counter
from data import DataPoint
import json
import pprint


def load_json_files(datasource_name_and_location, verbose=False):
    # Load data into memory (our data is small enough to safely fit in memory)
    scraped_pages = {}
    for name, filepath in datasource_name_and_location:
        with open(filepath) as json_data:
            scraped_pages[name] = json.load(json_data)

    if verbose:
        # View just one data point
        print("One post from NYT:")
        pprint.pprint([p['text'] for p in scraped_pages['newyorktimes'][110]['posts']])

    return scraped_pages


def build_dataset(scraped_pages, featurize_text, verbose=False):
    '''Build a featurized data set from scraped pages dictionary.
    '''
    dataset = []
    for datasource_name, all_datasource_data in scraped_pages.items():
        for scraped_page in all_datasource_data:
            if 'posts' in scraped_page:
                for post in scraped_page['posts']:
                    if 'text' in post:
                        text = post['text']
                        data_point = DataPoint(raw_data=text,
                                               featurize_function=featurize_text,
                                               klass=datasource_name)
                        dataset.append(data_point)

    if verbose:
        print("Dataset created. Class counts:\n {}".format(Counter([d.klass for d in dataset])))

    return dataset
