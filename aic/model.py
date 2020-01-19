import time
import tensorflow as tf
import utils.tools as utils
from utils.settings import config


class Model:

    # Constructor
    def __init__(self, model_path, data_file):
        # Load parameters
        self.embedding = utils.load_pkl(model_path + config['SNML']['embedding'])
        self.softmax_w = utils.load_pkl(model_path + config['SNML']['softmax_w'])
        self.n_vocab = self.embedding.shape[0]
        self.n_embedding = self.embedding.shape[1]
        self.n_context = self.softmax_w.shape[0]
        self.batch_size = 5000
        self.sum_log_likelihood = 0
        self.k = self.embedding.shape[0] * self.embedding.shape[1] + \
                 self.softmax_w.shape[0] * self.softmax_w.shape[1]

        # paths
        self.model_path = model_path
        self.filename = data_file

        # set computation
        self._set_computation(self.filename, batch_size=self.batch_size, epochs=1)

    def _set_computation(self, file_name, batch_size, epochs):
        # computation graph
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            # training data
            self.dataset = tf.data.experimental.make_csv_dataset(file_name,
                                                                 batch_size=batch_size,
                                                                 column_names=['input', 'output'],
                                                                 header=False,
                                                                 num_epochs=epochs)
            self.datum = self.dataset.make_one_shot_iterator().get_next()
            self.g_inputs, self.g_labels = self.datum['input'], self.datum['output']

            # embedding layer
            self.g_embedding = tf.get_variable("embedding", initializer=self.embedding)
            self.g_embed = tf.nn.embedding_lookup(self.g_embedding, self.g_inputs)

            # softmax layer
            self.g_softmax_w = tf.get_variable("softmax_w", initializer=self.softmax_w)

            # conditional probability of word given contexts
            self.g_mul = tf.transpose(tf.matmul(self.g_softmax_w, tf.transpose(self.g_embed)))
            self.g_logits = tf.exp(self.g_mul)
            self.g_sum_logits = tf.reduce_sum(self.g_logits, axis=1)
            self.indices = tf.stack([tf.range(tf.shape(self.g_labels)[0]), self.g_labels], axis=1)
            self.gather = tf.gather_nd(self.g_logits, self.indices)
            self.g_likelihood = self.gather / self.g_sum_logits
            self.g_sum_log_likelihood = tf.reduce_sum(tf.log(self.g_likelihood))

            # init variables
            self.g_init = tf.global_variables_initializer()

        # Tensorflow session
        self.sess = tf.Session(graph=self.train_graph)
        self.sess.run(self.g_init)

    def log_likelihood(self, print_step=1000):
        if self.sum_log_likelihood == 0:
            iteration = 1
            sum_log_likelihood = 0

            try:
                start = time.time()
                while True:
                    log_likelihood = self.sess.run(self.g_sum_log_likelihood)
                    sum_log_likelihood += log_likelihood

                    if iteration % print_step == 0:
                        end = time.time()
                        print("Iteration: {}".format(iteration),
                              "{:.4f} sec/ {} sample".format((end - start), self.batch_size * print_step))
                        start = time.time()

                    iteration += 1
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                print("Sum log likelihood: ", sum_log_likelihood)

            self.sum_log_likelihood = sum_log_likelihood

        return self.sum_log_likelihood

    def aic(self):
        return 2 * self.k - 2 * self.log_likelihood()
