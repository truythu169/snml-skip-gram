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


def get_loss_list_batch(m, d, batch_size=100):
    loss_list = []
    d = d.reshape(d.shape[0] // batch_size, -1, d.shape[1])

    for batch in d:
        w = batch[:, 0]
        c = batch[:, 1]
        loss_list.extend(m.get_loss_batch(w, c))

    return loss_list


if __name__ == "__main__":
    dims = [100, 110, 120, 130, 140, 150, 160, 170, 180]
    n_sample = 100000
    # read snml train file
    data = np.genfromtxt('../../../data/wiki/scope.csv', delimiter=',').astype(int)

    for dim in dims:
        print(dim)
        # full data
        model = Model('../../../output/wiki/20200126/1/train1/{}dim/step-90/'.format(dim),
                      '../../../data/text8/contexts/', n_context_sample=3000, learning_rate=0.1)

        loss_list = get_loss_list_batch(model, data[299999:n_sample + 299999])
        # loss_list = get_loss_list_batch(model, data[:n_sample])
        save_pkl(loss_list, 'cv_lines/cv_{}_dim.pkl'.format(dim), local=True)
