import time
import numpy as np
import utils.tools as utils
from utils.settings import config
from evaluation.word_analogy import WordAnalogy
from utils.embedding import Embedding
import csv
import os


class SkipGram:

    # Constructor
    def __init__(self, input_path, output_path, n_embedding, batch_size, epochs, n_sampled,
                 snml=False, snml_dir=''):
        self.n_embedding = n_embedding
        self.embedding = np.array([])
        self.data_path = input_path

        # create output directory
        self.output_dictionary = output_path + config['TRAIN']['output_dir'].format(n_embedding)
        if not os.path.exists(self.output_dictionary):
            os.makedirs(self.output_dictionary)

        if snml:
            self.snml_dir = snml_dir + config['TRAIN']['output_dir'].format(n_embedding)
            if not os.path.exists(self.snml_dir):
                os.makedirs(self.snml_dir)

        # sync with gcs
        utils.download_from_gcs(input_path + config['TRAIN']['int_to_vocab'])
        utils.download_from_gcs(input_path + config['TRAIN']['vocab_to_int'])
        utils.download_from_gcs(input_path + config['TRAIN']['context_dict'])
        utils.download_from_gcs(input_path + config['TRAIN']['cont_to_int'])
        utils.download_from_gcs(input_path + config['TRAIN']['train_data'])

        # read dictionaries
        self.int_to_vocab = utils.load_pkl(input_path + config['TRAIN']['int_to_vocab'])
        self.vocab_to_int = utils.load_pkl(input_path + config['TRAIN']['vocab_to_int'])
        self.int_to_cont = utils.load_pkl(input_path + config['TRAIN']['context_dict'])
        self.cont_to_int = utils.load_pkl(input_path + config['TRAIN']['cont_to_int'])
        self.n_vocab = len(self.int_to_vocab)
        self.n_context = len(self.int_to_cont)

        # computation parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_sampled = n_sampled
        random_seed = int(config['OTHER']['random_seed'])
        np.random.seed(random_seed)  # Set seed for weights
        self.E = [np.random.uniform(-0.8, 0.8, self.n_embedding) for _ in range(self.n_vocab)]
        self.F = np.random.uniform(-0.8, 0.8, (self.n_context, self.n_embedding))
        np.random.seed(random_seed)  # Set seed for anythings else (negative samples)

        # construct computation graph
        if snml:
            filename = self.data_path + config['TRAIN']['train_data_snml']
        else:
            filename = self.data_path + config['TRAIN']['train_data']
        self.snml = snml
        self.filename = filename

        # training data
        self.n_datums = utils.count_line(filename)
        self.n_batches = self.n_datums // self.batch_size

        # Context distribution
        self.context_distribution = utils.load_pkl(self.data_path + config['TRAIN']['context_dist'])
        self.context_distribution = self.context_distribution ** (3/4)
        self.context_distribution = self.context_distribution / sum(self.context_distribution)

        # Context sample look up table
        table_size = 100000000  # Length of the unigram table
        table = np.zeros(table_size, dtype=np.uint32)

        p = 0  # Cumulative probability
        i = 0
        for j in range(self.n_context):
            p += self.context_distribution[j]
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table
        self.pos_neg = [1] + [0 for _ in range(self.n_sampled)]

        # evaluator
        self.word_analogy = WordAnalogy(config['TRAIN']['question_file'])
        # self.word_analogy.set_top_words(config['TRAIN']['top_word_file'])

        # output file
        self.embedding_file = config['TRAIN']['embedding'].format(self.n_embedding, n_sampled, epochs, batch_size)

        print('Initialize Model with: {} samples, trying to run {} batches each epoch.'.format(self.n_datums,
                                                                                               self.n_batches))

    def train(self, learning_rate=0.01, print_step=1000, stop_threshold=0):
        losses = []
        aver_losses = []
        wa_scores = []
        if print_step == 0:
            print_step = self.n_batches

        for _ in range(self.epochs):
            iteration = 0
            start = time.time()

            with open(self.filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Print step
                    iteration += 1
                    if iteration % print_step == 0:
                        end = time.time()
                        print("Epochs: {}".format(_),
                              "Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(np.mean(losses)),
                              "{:.4f} sec/ {} sample".format((end - start), self.batch_size * print_step))
                        aver_losses.append(np.mean(losses))
                        losses = []
                        start = time.time()

                    # Print word analogy
                    if iteration % (print_step * 10) == 0:
                        eval = Embedding(np.array(self.E), self.int_to_vocab, self.vocab_to_int)
                        wa_score = self.word_analogy.evaluate(eval, high_level_category=False, restrict_top_words=False)
                        wa_scores.append(wa_score['all'])

                        self.export_model(self.output_dictionary + 'step-{}/'.format(int(iteration)))

                    loss = self._train_one_sample(int(row[0]), int(row[1]), learning_rate)
                    losses.append(loss)

            eval = Embedding(np.array(self.E), self.int_to_vocab, self.vocab_to_int)
            wa_score = self.word_analogy.evaluate(eval, high_level_category=False, restrict_top_words=False)
            wa_scores.append(wa_score['all'])
            print('Epochs: {}, WA score: {}'.format(_, wa_score['all']))

            # Save step
            if _ % 5 == 0:
                self.export_model(self.output_dictionary + 'step-{}/'.format(int(_)))

        # export losses
        utils.save_pkl(aver_losses, self.output_dictionary + config['TRAIN']['loss_file'])
        utils.save_pkl(wa_scores, self.output_dictionary + config['TRAIN']['acc_file'])

    def sample_contexts(self):
        indices = np.random.randint(low=0, high=len(self.table), size=self.n_sampled)
        return [self.table[i] for i in indices]

    def _train_one_sample(self, w, c, learning_rate=0.001):
        neg = self.sample_contexts()

        # Forward propagation
        e = self.E[w]
        labels = [c] + neg
        f_labels = self.F[labels].copy()
        a = np.dot(e, f_labels.T).reshape(-1, 1)
        p = utils.sigmoid(a)

        if p[0] == 0:
            print(w, c, a, utils.sigmoid(np.dot(e, self.F[labels].T)))
        for i in p[1:]:
            if i == 1:
                print(w, c, a, utils.sigmoid(np.dot(e, self.F[labels].T)))

        # Loss
        loss = -(np.log(p[0]) + np.sum(np.log(1-p[1:])))

        # Back propagation
        p[0] = p[0] - 1
        self.F[labels] -= learning_rate * p * np.tile(e, (len(p), 1))
        self.E[w] -= learning_rate * np.sum(p * f_labels, axis=0)

        return loss

    def export_embedding(self, filename='default'):
        # write embedding result to file
        if filename == 'default':
            filename = self.output_dictionary + self.embedding_file
        output = open(filename, 'w')
        for i in range(self.n_vocab):
            text = self.int_to_vocab[i]
            for j in self.E[i]:
                text += ' %f' % j
            text += '\n'
            output.write(text)

        output.close()

        # sync with to gcs
        utils.upload_to_gcs(filename, force_update=True)

    def export_model(self, out_dir='default'):
        if out_dir == 'default':
            out_dir = self.output_dictionary

        utils.save_pkl(self.E, out_dir + config['TRAIN']['embedding_pkl'])
        utils.save_pkl(self.F, out_dir + config['TRAIN']['softmax_w_pkl'])
