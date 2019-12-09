import numpy as np
import tensorflow as tf
import utils.tools as utils
from utils.settings import config


class Model:

    # Constructor
    def __init__(self, model_path, context_path, n_neg_sample=200, n_context_sample=1000,
                 learning_rate=0.0004, random_seed=1234):
        # Load parameters
        self.embedding = utils.load_pkl(model_path + config['SNML']['embedding'])
        self.softmax_w = utils.load_pkl(model_path + config['SNML']['softmax_w'])
        self.softmax_b = utils.load_pkl(model_path + config['SNML']['softmax_b'])
        self.n_vocab = self.embedding.shape[0]
        self.n_embedding = self.embedding.shape[1]
        self.n_context = self.softmax_w.shape[0]
        self.n_neg_sample = n_neg_sample
        self.context_path = context_path
        self.n_context_sample = n_context_sample
        self.scope = 0
        self.learning_rate = learning_rate

        # paths
        self.data_path = model_path

        # Set uniform distribution as default for context sampling
        self.contexts = []

        # set computation
        self._set_computation(random_seed)

    def _set_computation(self, random_seed):
        # computation graph
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            # set random seed
            tf.set_random_seed(random_seed)

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
            self.g_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.g_cost)
            # self.g_optimizer_one = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.g_cost)

            # conditional probability of word given contexts
            self.g_mul = tf.transpose(tf.matmul(self.g_softmax_w, tf.transpose(self.g_embed)))
            self.g_logits = tf.reshape(tf.exp(self.g_mul + self.g_softmax_b), [-1])
            self.g_sum_logits = tf.reduce_sum(self.g_logits)
            self.g_prob = tf.gather(self.g_logits, tf.reshape(self.g_labels, [-1])) / self.g_sum_logits

            # init variables
            self.g_init = tf.global_variables_initializer()

            # reset weights
            self.g_reset_embedding = self.g_embedding.assign(self.g_d_embedding)
            self.g_reset_softmax_w = self.g_softmax_w.assign(self.g_d_softmax_w)
            self.g_reset_softmax_b = self.g_softmax_b.assign(self.g_d_softmax_b)

            # update default weights
            self.g_update_d_embedding = self.g_d_embedding.assign(self.g_embedding)
            self.g_update_d_softmax_w = self.g_d_softmax_w.assign(self.g_softmax_w)
            self.g_update_d_softmax_b = self.g_d_softmax_b.assign(self.g_softmax_b)

        # Tensorflow session
        self.sess = tf.Session(graph=self.train_graph)
        self.sess.run(self.g_init)

    def get_neg_prob(self, word, context):
        feed = {self.g_inputs: [word],
                self.g_labels: [[context]]}

        train_loss = self.sess.run(self.g_cost, feed_dict=feed)

        return np.exp(-train_loss)

    def get_prob(self, word, context):
        feed = {self.g_inputs: [word],
                self.g_labels: [[context]]}

        prob = self.sess.run(self.g_prob, feed_dict=feed)

        return prob

    def snml_length(self, word, context, epochs=10):
        prob_sum = 0

        # Update all other context
        for i in range(self.n_context):
            prob = self.train_one_sample(word, i, epochs, update_weight=False)
            prob_sum += prob

        # Update true context and save weights
        prob = self.train_one_sample(word, context, epochs, update_weight=True)

        snml_length = - np.log(prob / prob_sum)
        return snml_length

    def snml_length_sampling(self, word, context, epochs=10):
        sample_contexts = self._sample_contexts(from_file=False)
        sample_prob = 1 / self.n_context
        prob_sum = 0

        # Update all other context
        for i in range(self.n_context_sample):
            c = sample_contexts[i]
            prob = self.train_one_sample(word, c, epochs, update_weight=False)
            prob_sum += prob

        prob_sum = prob_sum / (sample_prob * self.n_context_sample)

        # Update true context and save weights
        prob = self.train_one_sample(word, context, epochs, update_weight=True)

        snml_length = - np.log(prob / prob_sum)
        self.scope += 1
        return snml_length

    def train_one_sample(self, word, context, epochs=20, update_weight=False):
        # train weights
        for e in range(epochs):
            feed = {self.g_inputs: [word],
                    self.g_labels: [[context]]}

            train_loss, _ = self.sess.run([self.g_cost, self.g_optimizer], feed_dict=feed)

        # probability of word given contexts
        feed = {self.g_inputs: [word], self.g_labels: [[context]]}
        p = self.sess.run(self.g_prob, feed_dict=feed)

        # update weights (update default weights nodes in graph)
        if update_weight:
            self.sess.run(self.g_update_d_embedding)
            self.sess.run(self.g_update_d_softmax_w)
            self.sess.run(self.g_update_d_softmax_b)
        else:
            self.sess.run(self.g_reset_embedding)
            self.sess.run(self.g_reset_softmax_w)
            self.sess.run(self.g_reset_softmax_b)

        return p

    def _sample_contexts(self, from_file=True):
        if not from_file:
            samples = utils.sample_context_uniform(self.n_context, self.n_context_sample)
            print(samples[:10])
            return samples

        # Sample contexts
        if self.scope + 1 > len(self.contexts):
            for i in range(self.scope + 1 - len(self.contexts)):
                samples = utils.sample_context_uniform(self.n_context, self.n_context_sample)
                self.contexts.append(samples)

            # Save result back to pkl
            # print('Uploading sample context file, scope: ', self.scope)
            # utils.save_pkl(self.contexts, self.sample_contexts_file_name)

        return self.contexts[self.scope]
