import time
import numpy as np
import tensorflow as tf
import utils.tools as utils
from utils.settings import config
from evaluation.word_analogy import WordAnalogy
from utils.embedding import Embedding
import os


class SkipGram:

    # Constructor
    def __init__(self, input_path, output_path, n_embedding, batch_size, epochs, n_sampled,
                 snml=False, snml_dir='', random_seed=1234):
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
        utils.download_from_gcs(input_path + config['TRAIN']['vocab_dict'])
        utils.download_from_gcs(input_path + config['TRAIN']['context_dict'])
        utils.download_from_gcs(input_path + config['TRAIN']['train_data'])

        # read dictionaries
        self.int_to_vocab = utils.load_pkl(input_path + config['TRAIN']['vocab_dict'])
        self.vocab_to_int = utils.load_pkl(input_path + config['TRAIN']['int_vocab_dict'])
        self.int_to_cont = utils.load_pkl(input_path + config['TRAIN']['context_dict'])
        self.n_vocab = len(self.int_to_vocab)
        self.n_context = len(self.int_to_cont)

        # computation graph
        self.train_graph = tf.Graph()
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_sampled = n_sampled

        # construct computation graph
        if snml:
            filename = self.data_path + config['TRAIN']['train_data_snml']
        else:
            filename = self.data_path + config['TRAIN']['train_data']
        self.snml = snml
        self._set_training_file(filename)
        self._set_computation(filename, random_seed)

        # training data
        self.n_datums = utils.count_line(filename)
        self.n_batches = self.n_datums // self.batch_size

        # evaluator
        self.word_analogy = WordAnalogy(config['TRAIN']['question_file'])
        self.word_analogy.set_top_words(config['TRAIN']['top_word_file'])

        # output file
        self.embedding_file = config['TRAIN']['embedding'].format(self.n_embedding, n_sampled, epochs, batch_size)

        print('Initialize Model with: {} samples, trying to run {} batches each epoch.'.format(self.n_datums,
                                                                                               self.n_batches))

    def _set_training_file(self, file_name):
        # with self.train_graph.as_default():
        return

    def _set_computation(self, file_name, random_seed=1234):
        with self.train_graph.as_default():
            # set random seed
            tf.set_random_seed(random_seed)

            # training data
            self.dataset = tf.data.experimental.make_csv_dataset(file_name,
                                                                 batch_size=self.batch_size,
                                                                 column_names=['input', 'output'],
                                                                 header=False,
                                                                 num_epochs=self.epochs)
            self.datum = self.dataset.make_one_shot_iterator().get_next()
            self.inputs, self.labels = self.datum['input'], self.datum['output']

            # embedding layer
            self.embedding_g = tf.Variable(tf.random_uniform((self.n_vocab, self.n_embedding), -1, 1))
            self.embed = tf.nn.embedding_lookup(self.embedding_g, self.inputs)

            # softmax layer
            self.softmax_w_g = tf.Variable(tf.truncated_normal((self.n_context, self.n_embedding)))
            self.softmax_b_g = tf.Variable(tf.zeros(self.n_context))

            # Calculate the loss using negative sampling
            self.labels = tf.reshape(self.labels, [-1, 1])
            self.loss = tf.nn.sampled_softmax_loss(
                weights=self.softmax_w_g,
                biases=self.softmax_b_g,
                labels=self.labels,
                inputs=self.embed,
                num_sampled=self.n_sampled,
                num_classes=self.n_context)

            self.cost = tf.reduce_mean(self.loss)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

            # full loss
            logits = tf.matmul(self.embed, tf.transpose(self.softmax_w_g))
            logits = tf.nn.bias_add(logits, self.softmax_b_g)
            labels_one_hot = tf.one_hot(self.labels, self.n_context)
            self.full_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels_one_hot,
                logits=logits)
            self.full_cost = tf.reduce_mean(self.full_loss)
            self.full_optimizer = tf.train.AdamOptimizer().minimize(self.full_cost)

            # context distribution
            self.g_prob = tf.nn.softmax(logits)

            # init variables
            self.g_init = tf.global_variables_initializer()

        # Tensorflow session
        self.sess = tf.Session(graph=self.train_graph)
        self.sess.run(self.g_init)

    def _continue_graph(self):
        with self.train_graph.as_default():
            # continue weights
            self.assign_embedding = self.embedding_g.assign(self.embedding)
            self.assign_softmax_w = self.softmax_w_g.assign(self.softmax_w)
            self.assign_softmax_b = self.softmax_b_g.assign(self.softmax_b)

    def train(self, print_step=1000, stop_threshold=0):
        iteration = 1
        loss = 0
        losses = []
        epoch_sum_loss = 0.
        last_epoch_loss = 999999.
        wa_scores = []
        if print_step == 0:
            print_step = self.n_batches

        try:
            start = time.time()
            while True:
                # train_loss, _ = self.sess.run([self.full_cost, self.full_optimizer])
                train_loss, _ = self.sess.run([self.cost, self.optimizer])
                loss += train_loss
                epoch_sum_loss += train_loss
                losses.append(train_loss)

                if iteration % print_step == 0:
                    end = time.time()
                    print("Iteration: {}".format(iteration),
                          "Avg. Training loss: {:.4f}".format(loss / print_step),
                          "{:.4f} sec/ {} sample".format((end - start), self.batch_size * print_step))
                    loss = 0
                    start = time.time()

                if iteration % self.n_batches == 0:
                    epoch_loss = epoch_sum_loss / self.n_batches
                    epoch_sum_loss = 0
                    epoch_loss_diff = np.abs(epoch_loss - last_epoch_loss)
                    print('Epochs {} loss: {}'.format(iteration / self.n_batches, epoch_loss))

                    # word analogy score
                    embedding = self.sess.run(self.embedding_g)
                    eval = Embedding(embedding, self.int_to_vocab, self.vocab_to_int)
                    wa_score = self.word_analogy.evaluate(eval, high_level_category=False, restrict_top_words=False)
                    wa_scores.append(wa_score['all'])

                    # stop criteria
                    if epoch_loss_diff < stop_threshold:
                        self.epochs = iteration / self.n_batches
                        # output file
                        self.embedding_file = config['TRAIN']['embedding'].format(self.n_embedding, self.n_sampled,
                                                                                  int(self.epochs), self.batch_size)
                        print('Loss diff: {}, stop training.'.format(epoch_loss_diff))
                        print(self.output_dictionary + self.embedding_file)
                        break

                    last_epoch_loss = epoch_loss

                iteration += 1
        except tf.errors.OutOfRangeError:
            print("End of dataset")

        # export embedding matrix
        self.embedding = self.sess.run(self.embedding_g)
        self.softmax_w = self.sess.run(self.softmax_w_g)
        self.softmax_b = self.sess.run(self.softmax_b_g)

        # export losses
        utils.save_pkl(losses, self.output_dictionary + config['TRAIN']['loss_file'])
        utils.save_pkl(wa_scores, self.output_dictionary + config['TRAIN']['acc_file'])

    def export_embedding(self, filename='default'):
        # write embedding result to file
        if filename == 'default':
            filename = self.output_dictionary + self.embedding_file
        output = open(filename, 'w')
        for i in range(self.embedding.shape[0]):
            text = self.int_to_vocab[i]
            for j in self.embedding[i]:
                text += ' %f' % j
            text += '\n'
            output.write(text)

        output.close()

        # sync with to gcs
        utils.upload_to_gcs(filename, force_update=True)

    def export_model(self, out_dir='default'):
        if out_dir == 'default':
            out_dir = self.output_dictionary

        utils.save_pkl(self.embedding, out_dir + config['TRAIN']['embedding_pkl'])
        utils.save_pkl(self.softmax_w, out_dir + config['TRAIN']['softmax_w_pkl'])
        utils.save_pkl(self.softmax_b, out_dir + config['TRAIN']['softmax_b_pkl'])
