from multiprocessing import Pool
import numpy as np
import utils.tools as utils
import math as ma
import multiprocessing


class Model:

    def __init__(self, data_path, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.E = utils.load_pkl(data_path + 'embedding.pkl')
        self.C = utils.load_pkl(data_path + 'softmax_w.pkl')
        self.b = utils.load_pkl(data_path + 'softmax_b.pkl')
        self.V = self.E.shape[0]
        self.K = self.E.shape[1]
        self.V_dash = self.C.shape[0]
        self.data_path = data_path

        # adam optimizer initialize
        self.t = 256040
        self.t_default = 256040
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = learning_rate
        self.epsilon = epsilon
        self.beta1_t = beta1 ** self.t
        self.beta2_t = beta2 ** self.t

        # initialize things
        self.mE_t = np.zeros((self.V, self.K))
        self.mC_t = np.zeros((self.V_dash, self.K))
        self.mb_t = np.zeros(self.V_dash)
        self.vE_t = np.zeros((self.V, self.K))
        self.vC_t = np.zeros((self.V_dash, self.K))
        self.vb_t = np.zeros(self.V_dash)

    def get_prob(self, word, context):
        # forward propagation
        e = self.E[word]  # K dimensions vector
        z = np.dot(e, self.C.T) + self.b
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z)
        y = exp_z / sum_exp_z

        return y[context]

    def get_neg_prob(self, word, context, neg_size=200):
        neg = utils.sample_negative(neg_size, {context}, vocab_size=self.V_dash)

        # forward propagation
        e = self.E[word]  # K dimensions vector
        labels = [context] + neg
        z = np.dot(e, self.C[labels].T) + self.b[labels]
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z)
        prob = exp_z[0] / sum_exp_z

        return prob

    def snml_length(self, word, context, epochs=20, neg_size=200):
        prob_sum = 0
        iteration = 0

        # Update all other context
        for c in range(self.V_dash):
            if c != context:
                iteration += 1
                prob, losses = self.train(word, c, epochs, neg_size)

                prob_sum += prob

        # Update true context and save weights
        prob, losses = self.train(word, context, epochs, neg_size)
        prob_sum += prob
        snml_length = - np.log(prob / prob_sum)
        return snml_length

    def snml_length_sampling(self, word, context, epochs=20, neg_size=200, n_context_sample=600):
        sample_contexts, sample_contexts_prob = utils.sample_contexts(n_context_sample, self.t - self.t_default)
        prob_sum = 0
        probs = []

        # Update all other context
        for i in range(n_context_sample):
            c = sample_contexts[i]
            c_prob = sample_contexts_prob[i]

            prob = self.train(word, c, epochs, neg_size)
            prob_sum += prob / c_prob
            probs.append(prob)
        prob_sum = prob_sum / n_context_sample

        # Update true context and save weights
        prob = self.train(word, context, epochs, neg_size, update_weights=True)
        snml_length = - np.log(prob / prob_sum)

        return snml_length, probs

    def snml_length_sampling_multiprocess(self, word, context, epochs=20, neg_size=200, n_context_sample=600):
        sample_contexts, sample_contexts_prob = utils.sample_contexts(n_context_sample, self.t - self.t_default)

        # implement pools
        job_args = [(word, c, epochs, neg_size) for c in sample_contexts]
        p = Pool(multiprocessing.cpu_count())
        probs = p.map(self._train_job, job_args)
        p.close()
        p.join()

        # gather sum of probabilities
        prob_sum = 0
        for i in range(n_context_sample):
            prob_sum += probs[i] / sample_contexts_prob[i]
        prob_sum = prob_sum / n_context_sample

        # Update true context and save weights
        prob = self.train(word, context, epochs, neg_size, update_weights=True)
        snml_length = - np.log(prob / prob_sum)

        return snml_length, probs

    def _train_job(self, args):
        return self.train(*args)

    def train(self, w, c, epochs=20, neg_size=200, update_weights=False):
        if update_weights:
            prob = self._train_update_neg_adam(w, c, epochs, neg_size)
        else:
            # copy parameters
            e = self.E[w].copy()
            C_train = self.C.copy()
            b_train = self.b.copy()

            me_train = self.mE_t[w].copy()
            mC_train = self.mC_t.copy()
            mb_train = self.mb_t.copy()

            ve_train = self.vE_t[w].copy()
            vC_train = self.vC_t.copy()
            vb_train = self.vb_t.copy()

            t_train = self.t
            beta1_train = self.beta1_t
            beta2_train = self.beta2_t

            prob = self._train_neg_adam(e, c, epochs, neg_size, C_train, b_train, me_train, mC_train, mb_train, ve_train,
                                        vC_train, vb_train, t_train, beta1_train, beta2_train)

        return prob

    def _train_neg_adam(self, e, c, epochs, neg_size, C_train, b_train, me_train, mC_train, mb_train, ve_train,
                        vC_train, vb_train, t_train, beta1_train, beta2_train):
        # start epochs
        for i in range(epochs):
            neg = utils.sample_negative(neg_size, {c}, vocab_size=self.V_dash)

            # forward propagation
            labels = [c] + neg
            z = np.dot(e, C_train[labels].T) + b_train[labels]
            exp_z = np.exp(z)
            sum_exp_z = np.sum(exp_z)

            # back propagation
            dz = exp_z / sum_exp_z
            dz[0] -= 1  # for true label
            dz = dz / 10000
            dC = np.dot(dz.reshape(-1, 1), e.reshape(1, -1))
            db = dz
            dE = np.dot(dz.reshape(1, -1), C_train[labels]).reshape(-1)

            # adam step
            t_train = t_train + 1
            beta1_train = beta1_train * self.beta1
            beta2_train = beta2_train * self.beta2

            # adam things
            lr = self.lr * ma.sqrt(1 - beta2_train) / (1 - beta1_train)
            mE = self.beta1 * me_train + (1 - self.beta1) * dE
            mC = self.beta1 * mC_train[labels] + (1 - self.beta1) * dC
            mb = self.beta1 * mb_train[labels] + (1 - self.beta1) * db
            vE = self.beta2 * ve_train + (1 - self.beta2) * dE * dE
            vC = self.beta2 * vC_train[labels] + (1 - self.beta2) * dC * dC
            vb = self.beta2 * vb_train[labels] + (1 - self.beta2) * db * db

            # update weights
            e -= lr * mE / (np.sqrt(vE + self.epsilon))
            C_train[labels] -= lr * mC / (np.sqrt(vC + self.epsilon))
            b_train[labels] -= lr * mb / (np.sqrt(vb + self.epsilon))

            # save status
            me_train = mE
            mC_train[labels] = mC
            mb_train[labels] = mb
            ve_train = vE
            vC_train[labels] = vC
            vb_train[labels] = vb

        # get probability
        neg = utils.sample_negative(neg_size, {c}, vocab_size=self.V_dash)
        labels = [c] + neg
        z = np.dot(e, C_train[labels].T) + b_train[labels]
        exp_z = np.exp(z)
        prob = exp_z[0] / np.sum(exp_z)

        return prob

    def _train_update_neg_adam(self, w, c, epochs, neg_size):
        for i in range(epochs):
            neg = utils.sample_negative(neg_size, {c}, vocab_size=self.V_dash)

            # forward propagation
            e = self.E[w]
            labels = [c] + neg
            z = np.dot(e, self.C[labels].T) + self.b[labels]
            exp_z = np.exp(z)
            sum_exp_z = np.sum(exp_z)

            # back propagation
            dz = exp_z / sum_exp_z
            dz[0] -= 1  # for true label
            dz = dz / 100000
            dC = np.dot(dz.reshape(-1, 1), e.reshape(1, -1))
            db = dz
            dE = np.dot(dz.reshape(1, -1), self.C[labels]).reshape(-1)

            # adam step
            self.t = self.t + 1
            self.beta1_t = self.beta1_t * self.beta1
            self.beta2_t = self.beta2_t * self.beta2

            # adam things
            lr = self.lr * ma.sqrt(1 - self.beta2_t) / (1 - self.beta1_t)
            mE = self.beta1 * self.mE_t[w] + (1 - self.beta1) * dE
            mC = self.beta1 * self.mC_t[labels] + (1 - self.beta1) * dC
            mb = self.beta1 * self.mb_t[labels] + (1 - self.beta1) * db
            vE = self.beta2 * self.vE_t[w] + (1 - self.beta2) * dE * dE
            vC = self.beta2 * self.vC_t[labels] + (1 - self.beta2) * dC * dC
            vb = self.beta2 * self.vb_t[labels] + (1 - self.beta2) * db * db

            # update weights
            self.E[w] -= lr * mE / (np.sqrt(vE + self.epsilon))
            self.C[labels] -= lr * mC / (np.sqrt(vC + self.epsilon))
            self.b[labels] -= lr * mb / (np.sqrt(vb + self.epsilon))

            # save status
            self.mE_t[w] = mE
            self.mC_t[labels] = mC
            self.mb_t[labels] = mb
            self.vE_t[w] = vE
            self.vC_t[labels] = vC
            self.vb_t[labels] = vb

        # get probability
        neg = utils.sample_negative(neg_size, {c}, vocab_size=self.V_dash)
        labels = [c] + neg
        z = np.dot(self.E[w], self.C[labels].T) + self.b[labels]
        exp_z = np.exp(z)
        prob = exp_z[0] / np.sum(exp_z)

        return prob
