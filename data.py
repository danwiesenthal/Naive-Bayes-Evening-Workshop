import copy
import random


class DataPoint(object):

    def __init__(self, raw_data, featurize_function, klass=None):
        self.raw_data = raw_data
        self.featurize_function = featurize_function
        self.klass = klass
        self.featuredict = featurize_function(self.raw_data)


def split_dataset(dataset, fraction_train=0.8):
    ''' Copy our data here to avoid shuffling the original passed-in dataset
    list. This is expensive memory-wise (because we now have two copies of our
    dataset in memory rather than just one), but it keeps us from modifying
    the passed-in dataset outside this function, which is nice.  Overall there
    are better ways to structure code flow in terms of memory efficiency; this
    example is more for clarity of algorithm than efficiency of memory.
    '''
    dataset = copy.deepcopy(dataset)
    random.shuffle(dataset)
    split_index = int(len(dataset) * fraction_train)
    train_dataset, test_dataset = dataset[:split_index], dataset[split_index:]
    return train_dataset, test_dataset
