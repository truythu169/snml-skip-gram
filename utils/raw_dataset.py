import utils.tools as utils
import numpy as np
import os
from utils.settings import config


class RawDataset:

    def __init__(self, data_file, output_path):
        # Set output path
        self.output_path = output_path

        # read data from file
        print('Reading file: ', data_file)
        with open(data_file) as f:
            text = f.read()

        # about our data
        print('Parse data...')
        words, self.top_words = utils.preprocess(text)

        # word to int and int to word dictionaries, convert words to list of int
        # 0: vocab_to_int, 1: int_to_vocab, 2: cont_to_int, 3: int_to_cont
        dicts = utils.create_lookup_tables(words)
        int_words = [dicts[2][word] for word in words]

        # subsampling
        train_words = utils.get_train_words(int_words)

        # set class attributes
        self.n_vocab = len(dicts[1])
        self.n_context = len(dicts[3])
        self.vocab_to_int = dicts[0]
        self.int_to_vocab = dicts[1]
        self.cont_to_int = dicts[2]
        self.int_to_cont = dicts[3]
        self.words = train_words

        print("Total words: {}".format(len(words)))
        print("Unique words: {}".format(self.n_vocab))
        print("Unique context: {}".format(self.n_context))
        print("Data Prepared!")

        # Save dictionaries
        self.save_dicts()
        self.save_top_words()

    def save_dicts(self):
        # make directories
        dict_path = self.output_path + config['PREPROCESS']['output_dict_path']
        if not os.path.exists(dict_path):
            os.makedirs(dict_path)

        # Save dictionaries
        utils.save_pkl(self.vocab_to_int, dict_path + config['PREPROCESS']['vocab_to_int'])
        utils.save_pkl(self.int_to_vocab, dict_path + config['PREPROCESS']['int_to_vocab'])
        utils.save_pkl(self.cont_to_int, dict_path + config['PREPROCESS']['cont_to_int'])
        utils.save_pkl(self.int_to_cont, dict_path + config['PREPROCESS']['int_to_cont'])

    def save_top_words(self):
        output = open(self.output_path + '/top_{}_words.txt'.format(config['PREPROCESS']['n_top']), 'w')
        for word, count in self.top_words:
            output.write(word + '\n')
        output.close()

    def convert_context_to_word(self, context_id):
        word = self.int_to_cont[context_id]
        if word in self.vocab_to_int:
            return self.vocab_to_int[word]
        else:
            return False

    def get_target(self, words, idx, window_size=15):
        """ Get a list of words in a window around an index. """
        R = np.random.randint(1, window_size + 1)
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        target_words = set(words[start:idx] + words[idx + 1:stop + 1])

        return list(target_words)

    def get_batches(self, batch_size, window_size=15):
        """ Create a generator of word batches as a tuple (inputs, targets) """
        n_data = len(self.words)

        for idx in range(0, n_data, batch_size):
            x, y = [], []
            batch_upper_bound = idx + batch_size
            batch_upper_bound = n_data if (batch_upper_bound > n_data) else batch_upper_bound
            batch = self.words[idx:batch_upper_bound]

            for ii in range(len(batch)):
                batch_x = self.convert_context_to_word(batch[ii])
                if batch_x != False:
                    batch_y = self.get_target(batch, ii, window_size)
                    y.extend(batch_y)
                    x.extend([batch_x] * len(batch_y))
            yield x, y