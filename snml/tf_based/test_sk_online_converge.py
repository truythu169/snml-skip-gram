from snml.tf_based.model import Model
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import numpy as np


if __name__ == "__main__":
    epochs = 11
    dim = '500'

    # read snml train file
    data = np.genfromtxt('../../../data/text8/scope.csv', delimiter=',').astype(int)

    # full data
    model = Model('../../../output/text8/momentum/full/1/' + dim + 'dim/',
                  '../../../data/text8/contexts/', n_context_sample=3000, learning_rate=0.0004)

    p_full = []
    p_snml_b = []
    p_snml_a = []
    n_sample = 4509

    for i in range(n_sample):
        datum = data[i]
        w, c = data[i][0], data[i][1]
        w = int(w)
        c = int(c)

        pf = -np.log(model.get_prob(w, c))
        p_full.append(pf)

        if i % 100 == 0:
            print('{} th loop'.format(i))

    # SNML data
    tf.reset_default_graph()
    model = Model('../../../output/text8/momentum/snml/1/' + dim + 'dim/',
                  '../../../data/text8/contexts/', n_context_sample=3000, learning_rate=0.0004)

    for i in range(n_sample):
        datum = data[i]
        w, c = data[i][0], data[i][1]
        w = int(w)
        c = int(c)

        ps_b = -np.log(model.get_prob(w, c))
        p_snml_b.append(ps_b)

        if i % 100 == 0:
            print('{} th loop'.format(i))

    for i in range(n_sample):
        datum = data[i]
        w, c = data[i][0], data[i][1]
        w = int(w)
        c = int(c)

        # ps_a = -np.log(model.train_one_sample(w, c, epochs=epochs, update_weight=True))
        ps_a = 0
        p_snml_a.append(ps_a)

        if i % 100 == 0:
            print('{} th loop'.format(i))

    print('MAE before: ', mean_absolute_error(p_snml_b, p_full))
    print('MAE after: ', mean_absolute_error(p_snml_a, p_full))
    print(sum(p_full), sum(p_snml_a), sum(p_snml_b))
