from data import DataPoint
import json
import pprint


def load_scraped_json_files_into_DataPoint_objects(datasource_name_and_location, verbose=False):
    # Load data into memory (our data is small enough we can work with it in memory easily)
    scraped_pages = {}
    for datasource in datasource_name_and_location:
        with open(datasource[1]) as json_data:
            scraped_pages[datasource[0]] = json.load(json_data)

    if verbose:
        # View just one data point
        print("One post from NYT:")
        pprint.pprint([p['text'] for p in scraped_pages['newyorktimes'][110]['posts']])

    # Build a featurized data set
    dataset = []
    for datasource_name, all_datasource_data in scraped_pages.items():
        for scraped_page in all_datasource_data:
            if 'posts' in scraped_page:
                for post in scraped_page['posts']:
                    if 'text' in post:
                        text = post['text']
                        data_point = DataPoint(raw_data=text, klass=datasource_name)
                        dataset.append(data_point)

    # Return our dataset
    return dataset
