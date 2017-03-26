import numpy as np
import re
import itertools
from collections import Counter
from tensorflow.contrib import learn

class Data_Loader():
    def __init__(self, batch_size):

        self.batch_size = batch_size

        positive_data_file = "./data/rt-polaritydata/rt-polarity.pos"
        negative_data_file = './data/rt-polaritydata/rt-polarity.neg'

        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, "r").readlines())
        negative_examples = [s.strip() for s in negative_examples]
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [clean_str(sent) for sent in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)

        self.data_process(x_text, y)

    def data_process(self, x_text, y):

        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        dev_sample_percentage = 0.1
        dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
        self.x_train, self.x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        self.y_train, self.y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        self.sequence_length = self.x_train.shape[1]
        self.num_classes = self.y_train.shape[1]
        self.vocab_size = len(vocab_processor.vocabulary_)
        self.total_batches = int(len(self.x_train)*1.0/self.batch_size)
        
    def reset_pointer(self):
        self.pointer = 0

    def next_batch(self):
        

        begin = self.pointer*self.batch_size
        end = (self.pointer+1)*self.batch_size
        self.pointer+=1
        # self.x_train[]
        return self.x_train[begin:end], self.y_train[begin:end]


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

