from snml.tf_based.model import Model
from utils.tools import save_pkl
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
    dims = [40, 50, 50, 70, 80, 100]
    n_sample = 10000
    # read snml train file
    data = np.genfromtxt('../../../data/text8/scope2.csv', delimiter=',').astype(int)

    for dim in dims:
        # full data
        model = Model('../../../output/text8/20200114/snml/1/train2/{}dim/'.format(dim),
                      '../../../data/text8/contexts/', n_context_sample=3000, learning_rate=0.1)

        loss_list = get_loss_list(model, data[:n_sample])
        save_pkl(loss_list, 'cv_lines/cv_{}_dim.pkl'.format(dim))