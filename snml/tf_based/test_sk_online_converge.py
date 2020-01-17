from snml.tf_based.model import Model
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import numpy as np
import argparse
from utils.tools import save_pkl


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', default='55', type=str)
    parser.add_argument('--rate', default=0.0075, type=float)
    args = parser.parse_args()

    epochs = 1
    dim = args.dim
    learning_rate = args.rate
    full_datas = ['snml']
    test_datas = ['snml']
    n_sample = 10000
    # read snml train file
    data = np.genfromtxt('../../../data/text8/scope1.csv', delimiter=',').astype(int)

    mae_before = []
    mae_after = []
    loss_before = []
    loss_after = []
    # process = []

    for full_data, test_data in zip(full_datas, test_datas):
        print('full: ', full_data, 'test: ', test_data)

        # full data
        model = Model('../../../output/text8/20200107/{}/1/train2/{}dim/'.format(full_data, dim),
                      '../../../data/text8/contexts/', n_context_sample=3000, learning_rate=learning_rate)

        p_full = get_loss_list(model, data[:n_sample])

        # SNML data
        tf.reset_default_graph()
        model = Model('../../../output/text8/20200107/{}/1/train1/{}dim/'.format(test_data, dim),
                      '../../../data/text8/contexts/', n_neg_sample=3000, n_context_sample=3000, learning_rate=learning_rate)

        p_snml_b = get_loss_list(model, data[:n_sample])

        p_snml_a = []
        for i in range(n_sample):
            datum = data[i]
            w, c = int(data[i][0]), int(data[i][1])

            ps_a = -np.log(model.train_one_sample(w, c, epochs=epochs, update_weight=True))
            p_snml_a.append(ps_a)

            if i % 2000 == 0:
                print('{} th loop'.format(i))

        print('MAE before: ', mean_absolute_error(p_snml_b, p_full))
        print('MAE after: ', mean_absolute_error(p_snml_a, p_full))
        print(sum(p_full), sum(p_snml_a), sum(p_snml_b))

        mae_before.append(mean_absolute_error(p_snml_b, p_full))
        mae_after.append(mean_absolute_error(p_snml_a, p_full))
        loss_before.append(sum(p_full))
        loss_after.append(sum(p_snml_a))

    # save_pkl(process, 'cv_lines/zero_snml_{}dim.pkl'.format(dim), local=True)
    # save_pkl(p_snml_a, 'cv_lines/after_snml_{}dim.pkl'.format(dim), local=True)

    print('Before: ')
    print_array(mae_before)
    print_array(loss_before)

    print('After: ')
    print_array(mae_after)
    print_array(loss_after)
