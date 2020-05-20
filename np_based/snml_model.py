from multiprocessing import Pool
from utils.settings import config
import numpy as np
import utils.tools as utils
import multiprocessing


class Model:

    def __init__(self, model_path, context_path, learning_rate=0.001, n_negative_sample=15):
        self.E = utils.load_pkl(model_path + 'embedding.pkl', local=True)
        self.E = np.array(self.E)
        self.F = utils.load_pkl(model_path + 'softmax_w.pkl', local=True)
        self.n_vocab = len(self.E)
        self.d = self.F.shape[1]
        self.n_context = self.F.shape[0]
        self.data_path = model_path
        self.n_negative_sample = n_negative_sample
        self.scope = 0

        # Context distribution
        self.context_distribution = utils.load_pkl(context_path + config['TRAIN']['context_dist'])
        self.context_distribution = self.context_distribution ** (3 / 4)
        self.context_distribution = self.context_distribution / sum(self.context_distribution)

        # Optimizer initialize
        self.lr = learning_rate

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

        # Set random seed
        np.random.seed(int(config['OTHER']['random_seed']))

    def get_prob(self, word, context):
        # forward propagation
        e = self.E[word]
        a = np.dot(e, self.F[context].T)
        p = utils.sigmoid(a)

        return p

    def get_context_dis(self, word):
        # forward propagation
        e = self.E[word]
        a = np.dot(e, self.F.T).reshape(-1)
        p = utils.sigmoid(a)
        p = p / sum(p)

        return p

    def validation_loss(self, word, context):
        neg_samples = self.sample_contexts()

        # forward propagation
        e = self.E[word]
        a = np.dot(e, self.F[[context] + neg_samples].T)
        p = utils.sigmoid(a)

        # compute loss
        loss = - np.log(p[0]) - sum(np.log(1-p[1:]))

        return loss

    def sample_contexts(self):
        indices = np.random.randint(low=0, high=len(self.table), size=self.n_negative_sample)
        return [self.table[i] for i in indices]

    def snml_length(self, word, context, epochs=20):
        neg_samples = self.sample_contexts()
        labels = [context] + neg_samples
        e = self.E[word]
        F = self.F[labels]

        # Training for noise
        prob_sum = 0
        for i in range(1, len(labels)):
            prob_sum += self._train_noise(e.copy(), F.copy(), i, epochs)

        # Update true context and save weights
        prob = self._train(word, labels, 0, epochs, update_weights=True)
        snml_length = - np.log(prob / (prob_sum + prob))

        return snml_length

    def snml_length_multiprocess(self, word, context, epochs=20):
        neg_samples = self.sample_contexts()

        # implement pools
        labels = [context] + neg_samples
        job_args = [(word, labels, i, epochs) for i in range(1, len(labels))]
        p = Pool(multiprocessing.cpu_count() // 2)
        probs = p.map(self._train_job, job_args)
        p.close()
        p.join()

        # gather sum of probabilities
        prob_sum = sum(probs)

        # Update true context and save weights
        prob = self._train(word, labels, 0, epochs, update_weights=True)
        snml_length = - np.log(prob / (prob_sum + prob))

        return snml_length

    def train(self, w, context, epochs):
        neg_samples = self.sample_contexts()
        labels = [context] + neg_samples

        return self._train(w, labels, 0, epochs, update_weights=True)

    def _train_job(self, args):
        return self._train(*args)

    def _train(self, w, labels, pos_sample_index, epochs=20, update_weights=False):
        if update_weights:
            prob = self._train_pos(w, labels, pos_sample_index, epochs)
        else:
            # copy parameters
            e = self.E[w].copy()
            F = self.F[labels].copy()

            prob = self._train_noise(e, F, pos_sample_index, epochs)

        return prob

    def _train_noise(self, e, F, pos_sample_index, epochs):
        for _ in range(epochs):
            # Forward propagation
            a = np.dot(e, F.T).reshape(-1, 1)
            p = utils.sigmoid(a)

            # Back propagation
            p[pos_sample_index] = p[pos_sample_index] - 1
            e_ = e.copy()
            e -= self.lr * np.sum(p * F, axis=0)
            F -= self.lr * p * np.tile(e_, (len(p), 1))

        # forward propagation
        a = np.dot(e, F.T)
        p = utils.sigmoid(a)

        # compute joint probability
        prob = p[pos_sample_index] * np.prod(1 - np.delete(p, pos_sample_index))

        return prob

    def _train_pos(self, w, labels, pos_sample_index, epochs):
        for _ in range(epochs):
            # Forward propagation
            e = self.E[w].copy()
            a = np.dot(e, self.F[labels].T).reshape(-1, 1)
            p = utils.sigmoid(a)

            # Back propagation
            p[pos_sample_index] = p[pos_sample_index] - 1
            self.E[w] -= self.lr * np.sum(p * self.F[labels], axis=0)
            self.F[labels] -= self.lr * p * np.tile(e, (len(p), 1))

        # forward propagation
        a = np.dot(self.E[w], self.F[labels].T)
        p = utils.sigmoid(a)

        # compute joint probability
        prob = p[pos_sample_index] * np.prod(1 - np.delete(p, pos_sample_index))

        return prob
