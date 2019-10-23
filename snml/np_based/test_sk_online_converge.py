from snml.np_based.model_momentum import ModelMomentum
from sklearn.metrics import mean_absolute_error
from collections import Counter
import numpy as np


if __name__ == "__main__":
    # read snml train file
    data = np.genfromtxt('../../../data/text8/scope.csv', delimiter=',').astype(int)

    # unique data
    str_data = set()
    for datum in data:
        str_data.add(str(datum[0]) + ',' + str(datum[1]))
    word_counts = Counter(str_data)
    data = [word for word in str_data if word_counts[word] < 2]
    data.sort()

    # full data
    model_full = ModelMomentum('../../../output/text8/momentum/full/48epochs/1/50dim/',
                               '../../../data/text8/contexts/', n_context_sample=600, learning_rate=0.00004, beta=0.9)
    model_snml = ModelMomentum('../../../output/text8/momentum/snml/48epochs/1/50dim/',
                               '../../../data/text8/contexts/', n_context_sample=600, learning_rate=0.00004, beta=0.9)

    p_full = []
    p_snml_b = []
    p_snml_a = []
    percent_error = 0
    n_sample = 100

    for i in range(n_sample):
        datum = data[i]
        w, c = datum.split(',')
        w = int(w)
        c = int(c)

        ps_b = -np.log(model_snml.get_prob(w, c))
        ps_a = -np.log(model_snml.train(w, c, epochs=48, neg_size=3000, update_weights=True))
        pf = -np.log(model_full.get_prob(w, c))
        print(ps_b, ps_a, pf)

        p_full.append(pf)
        p_snml_a.append(ps_a)
        p_snml_b.append(ps_b)
        percent_error += abs(ps_a - pf) / pf

        if i % 100 == 0:
            print('{} th loop'.format(i))

    print('MAE before: ', mean_absolute_error(p_snml_b, p_full))
    print('MAE after: ', mean_absolute_error(p_snml_a, p_full))
    print('Mean percent error: ', (percent_error * 100 / n_sample))
