import time
import numpy as np
import tensorflow as tf
import utils.tools as utils
from utils.settings import config


class Model:

    # Constructor
    def __init__(self, model_path, sample_path, output_path, context_distribution_file, n_train_sample=1000,
                 n_neg_sample=200, n_context_sample=1000):
        # Load parameters
        self.embedding = utils.load_pkl(model_path + config['SNML']['embedding'])
        self.softmax_w = utils.load_pkl(model_path + config['SNML']['softmax_w'])
        self.softmax_b = utils.load_pkl(model_path + config['SNML']['softmax_b'])
        self.n_vocab = self.embedding.shape[0]
        self.n_embedding = self.embedding.shape[1]
        self.n_context = self.softmax_w.shape[0]
        self.n_neg_sample = n_neg_sample

        # paths
        self.data_path = model_path
        self.sample_path = sample_path
        self.output_path = output_path
        self.n_files = int(config['SNML']['n_files'])

        # sample data
        self.n_train_sample = n_train_sample
        self.words = []
        self.contexts = []
        self.epochs = 0
        self._set_training_sample(20)
        self.sample_contexts, self.sample_contexts_prob = utils.sample_contexts(context_distribution_file, n_context_sample)
        self.n_context_sample = n_context_sample

        # set computation
        self._set_computation()

    def _set_computation(self):
        # computation graph
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            # training data
            # input placeholders
            self.g_inputs = tf.placeholder(tf.int32, [None], name='inputs')
            self.g_labels = tf.placeholder(tf.int32, [None, None], name='labels')

            # default weights
            self.g_d_embedding = tf.get_variable("d_embedding", initializer=self.embedding)
            self.g_d_softmax_w = tf.get_variable("d_softmax_w", initializer=self.softmax_w)
            self.g_d_softmax_b = tf.get_variable("d_softmax_b", initializer=self.softmax_b)

            # embedding layer
            self.g_embedding = tf.get_variable("embedding", initializer=self.embedding)
            self.g_embed = tf.nn.embedding_lookup(self.g_embedding, self.g_inputs)

            # softmax layer
            self.g_softmax_w = tf.get_variable("softmax_w", initializer=self.softmax_w)
            self.g_softmax_b = tf.get_variable("softmax_b", initializer=self.softmax_b)

            # Calculate the loss using negative sampling
            self.g_labels = tf.reshape(self.g_labels, [-1, 1])
            self.g_loss = tf.nn.sampled_softmax_loss(
                weights=self.g_softmax_w,
                biases=self.g_softmax_b,
                labels=self.g_labels,
                inputs=self.g_embed,
                num_sampled=self.n_neg_sample,
                num_classes=self.n_context)

            # training operations
            self.g_cost = tf.reduce_mean(self.g_loss)
            self.g_optimizer = tf.train.AdamOptimizer().minimize(self.g_cost)
            self.g_optimizer_one = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.g_cost)

            # conditional probability of word given contexts
            # self.g_mul = tf.matmul(self.g_embed, tf.transpose(self.g_softmax_w))
            # self.g_logits = tf.reshape(tf.exp(self.g_mul + self.g_softmax_b), [-1])
            # self.g_sum_logits = tf.reduce_sum(self.g_logits)
            # self.g_prob = tf.gather(self.g_logits, tf.reshape(self.g_labels, [-1])) / self.g_sum_logits

            # init variables
            self.g_init = tf.global_variables_initializer()

            # reset weights
            self.g_reset_embedding = self.g_embedding.assign(self.g_d_embedding)
            self.g_reset_softmax_w = self.g_softmax_w.assign(self.g_d_softmax_w)
            self.g_reset_softmax_b = self.g_softmax_b.assign(self.g_d_softmax_b)

        # Tensorflow session
        self.sess = tf.Session(graph=self.train_graph)
        self.sess.run(self.g_init)

    def change_model(self, model_path):
        # Load parameters
        self.embedding = utils.load_pkl(model_path + config['SNML']['embedding'])
        self.softmax_w = utils.load_pkl(model_path + config['SNML']['softmax_w'])
        self.softmax_b = utils.load_pkl(model_path + config['SNML']['softmax_b'])
        self.n_vocab = self.embedding.shape[0]
        self.n_embedding = self.embedding.shape[1]
        self.n_context = self.softmax_w.shape[0]

        # paths
        self.data_path = model_path

        # set computation
        self._set_computation()

    def get_neg_prob(self, word, context):
        feed = {self.g_inputs: [word],
                self.g_labels: [[context]]}

        train_loss, _ = self.sess.run([self.g_cost, self.g_optimizer], feed_dict=feed)

        return np.exp(-train_loss)

    def snml_length(self, word, context, epochs=10):
        print('Start training for {} contexts ...'.format(self.n_context))
        prob_sum = 0
        iteration = 0

        # Update all other context
        start = time.time()
        for c in range(self.n_context):
            if c != context:
                iteration += 1
                prob = self._train_sample(word, c, epochs, update_weight=False)
                prob_sum += prob

                if iteration % 1000 == 0:
                    end = time.time()
                    print("Iteration: {}, ".format(iteration),
                          "{:.4f} sec".format(end - start))
                    start = time.time()

        # Update true context and save weights
        prob = self._train_sample(word, context, epochs, update_weight=True)
        prob_sum += prob
        snml_length = - np.log(prob / prob_sum)
        print('Finished!')
        return snml_length

    def snml_length_sampling(self, word, context, epochs=10):
        print('Start training for {} contexts ...'.format(self.n_context))
        prob_sum = 0
        iteration = 0

        # Update all other context
        start = time.time()
        for i in range(self.n_context_sample):
            c = self.sample_contexts[i]
            c_prob = self.sample_contexts_prob[i]

            iteration += 1
            prob = self._train_sample(word, c, epochs, update_weight=False)
            prob_sum += prob / c_prob

            if iteration % 100 == 0:
                end = time.time()
                print("Iteration: {}, ".format(iteration),
                      "{:.4f} sec".format(end - start))
                start = time.time()
        prob_sum = prob_sum / self.n_context_sample

        # Update true context and save weights
        prob = self._train_sample(word, context, epochs, update_weight=True)

        snml_length = - np.log(prob / prob_sum)
        print('Finished!')
        return snml_length

    def train(self, word, context, epochs=10, update_weight=True, train_one=False):
        if train_one:
            prob = self._train_one_sample(word, context, epochs, update_weight)
        else:
            prob = self._train_sample(word, context, epochs, update_weight)
        return prob

    def _train_sample(self, word, context, epochs=10, update_weight=False):
        self._set_training_sample(epochs)

        # train weights
        for e in range(epochs):
            words, contexts = self._get_sample_data(word, context, e)
            feed = {self.g_inputs: words,
                    self.g_labels: np.array(contexts)[:, None]}

            train_loss, _ = self.sess.run([self.g_cost, self.g_optimizer], feed_dict=feed)
            # print(train_loss)

        # estimate conditional probability of word given contexts
        # feed = {self.g_inputs: [word], self.g_labels: [[context]]}
        # p = self.sess.run(self.g_prob, feed_dict=feed)
        p = np.exp(-train_loss)

        # update weights
        if not update_weight:
            self.sess.run(self.g_reset_embedding)
            self.sess.run(self.g_reset_softmax_w)
            self.sess.run(self.g_reset_softmax_b)

        return p

    def _train_one_sample(self, word, context, epochs=20, update_weigh=False):
        self._set_training_sample(epochs)

        # train weights
        for e in range(epochs):
            feed = {self.g_inputs: [word],
                    self.g_labels: [[context]]}

            train_loss, _ = self.sess.run([self.g_cost, self.g_optimizer], feed_dict=feed)
            # print(train_loss)

        # estimate conditional probability of word given contexts
        # feed = {self.g_inputs: [word], self.g_labels: [[context]]}
        # p = self.sess.run(self.g_prob, feed_dict=feed)
        p = np.exp(-train_loss)

        # update weights
        if not update_weigh:
            self.sess.run(self.g_reset_embedding)
            self.sess.run(self.g_reset_softmax_w)
            self.sess.run(self.g_reset_softmax_b)

        return p

    def _set_training_sample(self, epochs):
        if epochs > self.epochs:
            for i in range(epochs - self.epochs):
                words, contexts = utils.sample_learning_data(self.sample_path, self.n_files, self.n_train_sample)
                self.words.append(words)
                self.contexts.append(contexts)
            self.epochs = epochs

    def _get_sample_data(self, word, context, epoch):
        # sampling sample from train data
        words = self.words[epoch] + [word]
        contexts = self.contexts[epoch] + [context]

        return words, contexts