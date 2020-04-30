from np_based.snml_model import Model
from sklearn.metrics import mean_absolute_error
import numpy as np
import argparse


def print_array(a):
    for e in a:
        print(e)


def get_loss_list(m, d):
    loss_list = []
    for datum in d:
        w, c = int(datum[0]), int(datum[1])
        prob = m.get_prob(w, c)
        loss_list.append(prob)

    return - np.log(loss_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', default='50', type=str)
    parser.add_argument('--rate', default=0.0066, type=float)
    args = parser.parse_args()

    epochs = 2
    dim = args.dim
    learning_rate = args.rate
    full_datas = ['snml']
    test_datas = ['snml']
    n_sample = 1000
    # read snml train file
    data = np.genfromtxt('../../data/text8/shufle/1/scope.csv', delimiter=',').astype(int)

    mae_before = []
    mae_after = []
    loss_before = []
    loss_after = []
    for full_data, test_data in zip(full_datas, test_datas):
        print('full: ', full_data, 'test: ', test_data)

        # full data
        model = Model('../../output/sgns/text8/2/train2/{}dim/'.format(dim),
                      '../../data/text8/', learning_rate=0.008)

        p_full = get_loss_list(model, data[:n_sample])

        # SNML data
        model = Model('../../output/sgns/text8/2/train1/{}dim/'.format(dim),
                      '../../data/text8/', learning_rate=0.008)

        p_snml_b = get_loss_list(model, data[:n_sample])

        for i in range(n_sample):
            datum = data[i]
            w, c = int(data[i][0]), int(data[i][1])

            ps_a = -np.log(model.train(w, c, epochs=epochs))

            if i % 2000 == 0:
                print('{} th loop'.format(i))

        p_snml_a = get_loss_list(model, data[:n_sample])

        print('MAE before: ', mean_absolute_error(p_snml_b, p_full))
        print('MAE after: ', mean_absolute_error(p_snml_a, p_full))
        print(sum(p_full), sum(p_snml_a), sum(p_snml_b))

        mae_before.append(mean_absolute_error(p_snml_b, p_full))
        mae_after.append(mean_absolute_error(p_snml_a, p_full))
        loss_before.append(sum(p_full))
        loss_after.append(sum(p_snml_a))

    print('Before: ')
    print_array(mae_before)
    print_array(loss_before)

    print('After: ')
    print_array(mae_after)
    print_array(loss_after)
