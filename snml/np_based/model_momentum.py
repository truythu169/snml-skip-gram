from multiprocessing import Pool
import numpy as np
import utils.tools as utils
import os
import multiprocessing


class ModelMomentum:

    def __init__(self, data_path, context_path, learning_rate=0.2, beta=0.9, n_context_sample=600):
        self.E = utils.load_pkl(data_path + 'embedding.pkl')
        self.C = utils.load_pkl(data_path + 'softmax_w.pkl')
        self.b = utils.load_pkl(data_path + 'softmax_b.pkl')
        self.V = self.E.shape[0]
        self.K = self.E.shape[1]
        self.V_dash = self.C.shape[0]
        self.data_path = data_path
        self.context_path = context_path
        self.n_context_sample = n_context_sample
        self.scope = 0

        # Load context distribution
        self.context_distribution = utils.load_pkl(context_path + 'context_distribution.pkl')
        # Check if sample context file exits
        self.sample_contexts_file_name = os.path.join(self.context_path, 'sample_contexts_{}.pkl'.format(n_context_sample))
        self.contexts = utils.load_pkl(self.sample_contexts_file_name)

        # Momentum hyper parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.vE = np.zeros((self.V, self.K))
        self.vC = np.zeros((self.V_dash, self.K))
        self.vb = np.zeros(self.V_dash)

    def get_prob(self, word, context):
        # forward propagation
        e = self.E[word]  # K dimensions vector
        z = np.dot(e, self.C.T) + self.b
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z)
        y = exp_z[context] / sum_exp_z

        return y

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

    def snml_length_sampling(self, word, context, epochs=20, neg_size=200):
        sample_contexts, sample_contexts_prob = self._sample_contexts(from_file=True)
        prob_sum = 0
        probs = []

        # Update all other context
        for i in range(self.n_context_sample):
            c = sample_contexts[i]
            c_prob = sample_contexts_prob[i]

            prob = self.train(word, c, epochs, neg_size)
            prob_sum += prob / c_prob
            probs.append(prob)
        prob_sum = prob_sum / self.n_context_sample

        # Update true context and save weights
        prob = self.train(word, context, epochs, neg_size, update_weights=True)
        snml_length = - np.log(prob / prob_sum)
        self.scope += 1

        return snml_length, probs

    def snml_length_sampling_multiprocess(self, word, context, epochs=20, neg_size=200):
        sample_contexts, sample_contexts_prob = self._sample_contexts(from_file=True)

        # implement pools
        job_args = [(word, c, epochs, neg_size) for c in sample_contexts]
        p = Pool(multiprocessing.cpu_count())
        probs = p.map(self._train_job, job_args)
        p.close()
        p.join()

        # gather sum of probabilities
        prob_sum = 0
        for i in range(self.n_context_sample):
            prob_sum += probs[i] / sample_contexts_prob[i]
        prob_sum = prob_sum / self.n_context_sample

        # Update true context and save weights
        prob = self.train(word, context, epochs, neg_size, update_weights=True)
        snml_length = - np.log(prob / prob_sum)
        self.scope += 1

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

            ve_train = self.vE[w].copy()
            vC_train = self.vC.copy()
            vb_train = self.vb.copy()

            prob = self._train_neg_adam(e, c, epochs, neg_size, C_train, b_train, ve_train, vC_train, vb_train)

        return prob

    def _train_neg_adam(self, e, c, epochs, neg_size, C_train, b_train, ve_train, vC_train, vb_train):
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
            dz = dz
            dC = np.dot(dz.reshape(-1, 1), e.reshape(1, -1))
            db = dz
            dE = np.dot(dz.reshape(1, -1), C_train[labels]).reshape(-1)

            # adam step
            vE = self.beta * ve_train + dE
            vC = self.beta * vC_train[labels] + dC
            vb = self.beta * vb_train[labels] + db

            # update weights
            e -= self.learning_rate * vE
            C_train[labels] -= self.learning_rate * vC
            b_train[labels] -= self.learning_rate * vb

            # save status
            ve_train = vE
            vC_train[labels] = vC
            vb_train[labels] = vb

        # get probability
        z = np.dot(e, C_train.T) + b_train
        exp_z = np.exp(z)
        prob = exp_z[c] / np.sum(exp_z)

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
            dz = dz
            dC = np.dot(dz.reshape(-1, 1), e.reshape(1, -1))
            db = dz
            dE = np.dot(dz.reshape(1, -1), self.C[labels]).reshape(-1)

            # momentum things
            vE = self.beta * self.vE[w] + dE
            vC = self.beta * self.vC[labels] + dC
            vb = self.beta * self.vb[labels] + db

            # update weights
            self.E[w] -= self.learning_rate * vE
            self.C[labels] -= self.learning_rate * vC
            self.b[labels] -= self.learning_rate * vb

            # save status
            self.vE[w] = vE
            self.vC[labels] = vC
            self.vb[labels] = vb

        return self.get_prob(w, c)

    def _sample_contexts(self, from_file=True):
        if not from_file:
            samples = utils.sample_context(self.context_distribution, self.n_context_sample)
            return samples

        # Sample contexts
        if self.scope + 1 > len(self.contexts):
            for i in range(self.scope + 1 - len(self.contexts)):
                samples = utils.sample_context(self.context_distribution, self.n_context_sample)
                self.contexts.append(samples)

            # Save result back to pkl
            print('Uploading sample context file, scope: ', self.scope)
            utils.save_pkl(self.contexts, self.sample_contexts_file_name)

        return self.contexts[self.scope]
