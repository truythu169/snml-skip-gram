from snml.tf_based.model import Model
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import numpy as np


def print_array(a):
    for e in a:
        print(e)


if __name__ == "__main__":
    epochs = 2
    dim = '200'
    learning_rate = 0.00165
    full_datas = ['490000/', '495000/', '']
    test_datas = ['485000/', '490000/', '495000/']
    mae_before = []
    mae_after = []
    loss_before = []
    loss_after = []

    for full_data, test_data in zip(full_datas, test_datas):
        print('full: ', full_data, 'test: ', test_data)

        # read snml train file
        data = np.genfromtxt('../../notebooks/output/50-context-500000-data-18-questions/{}scope.csv'.format(test_data), delimiter=',').astype(int)

        # full data
        model = Model('../../notebooks/output/50-context-500000-data-18-questions/{}model/{}dim/'.format(full_data, dim),
                      '../../../data/text8/contexts/', n_context_sample=3000, learning_rate=learning_rate)

        p_full = []
        p_snml_b = []
        p_snml_a = []
        n_sample = 5000

        for i in range(n_sample):
            datum = data[i]
            w, c = data[i][0], data[i][1]
            w = int(w)
            c = int(c)

            pf = -np.log(model.get_prob(w, c))
            p_full.append(pf)

        # SNML data
        tf.reset_default_graph()
        model = Model('../../notebooks/output/50-context-500000-data-18-questions/{}model/{}dim/'.format(test_data, dim),
                      '../../../data/text8/contexts/', n_neg_sample=50, n_context_sample=3000, learning_rate=learning_rate)

        for i in range(n_sample):
            datum = data[i]
            w, c = data[i][0], data[i][1]
            w = int(w)
            c = int(c)

            ps_b = -np.log(model.get_prob(w, c))
            p_snml_b.append(ps_b)

        for i in range(n_sample):
            datum = data[i]
            w, c = data[i][0], data[i][1]
            w = int(w)
            c = int(c)

            ps_a = -np.log(model.train_one_sample(w, c, epochs=epochs, update_weight=True))
            p_snml_a.append(ps_a)

            if i % 2000 == 0:
                print('{} th loop'.format(i))

        print('MAE before: ', mean_absolute_error(p_snml_b, p_full))
        print('MAE after: ', mean_absolute_error(p_snml_a, p_full))
        print(sum(p_full), sum(p_snml_a), sum(p_snml_b))

        mae_before.append(mean_absolute_error(p_snml_b, p_full))
        mae_after.append(mean_absolute_error(p_snml_a, p_full))
        loss_before.append(sum(p_full)[0])
        loss_after.append(sum(p_snml_a)[0])

    print('MAE before: ')
    print_array(mae_before)
    print('MAE after: ')
    print_array(mae_after)
    print('Loss before: ')
    print_array(loss_before)
    print('Loss after: ')
    print_array(loss_after)
