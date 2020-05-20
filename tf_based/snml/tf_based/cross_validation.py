from tf_based.snml.tf_based.model import Model
import tensorflow as tf
import numpy as np


def print_array(a):
    for e in a:
        print(e)


def get_loss_list(m, d):
    loss_list = []

    for datum in d:
        w, c = int(datum[0]), int(datum[1])
        loss_list.append(m.get_loss(w, c))

    return loss_list


if __name__ == "__main__":
    dims = [50, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300]
    n_sample = 30000
    # read snml train file
    data = np.genfromtxt('../../../data/wiki/scope.csv', delimiter=',').astype(int)

    cvs = []
    for dim in dims:
        # full data
        tf.reset_default_graph()
        model = Model('../../../output/wiki/20200126/1/train1/{}dim/step-90/'.format(dim),
                      '../../../data/wiki/contexts/', n_context_sample=3000, learning_rate=0.1)

        p_full = get_loss_list(model, data[299999:n_sample + 299999])
        cvs.append(sum(p_full))

    for dim, cv in zip(dims, cvs):
        print('Dim {}: {}'.format(dim, cv))
