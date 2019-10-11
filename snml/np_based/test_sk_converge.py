from snml.np_based.model import Model
import numpy as np


if __name__ == "__main__":
    model = Model('../../../output/text8/snml/3000samples/31epochs/60dim/',
                  '../../../data/text8/contexts/', n_context_sample=600)

    words = [6581, 93, 4519, 506, 1687, 11469, 1188, 11469, 102, 8229, 2036]
    contexts = [390, 1172, 1545, 22, 72, 1659, 4363, 41, 9023, 693]

    for i in range(len(words)):
        word = words[i]
        context = contexts[i]

        p_sum = []
        for j in range(100):
            p = model.train(word, context, epochs=31, neg_size=3000)
            p_sum.append(-np.log(p))

        print('Mean: {:.4f} Min: {:.4f} Max: {:.4f} std: {:.4f}'.format(np.mean(p_sum), min(p_sum), max(p_sum), np.std(p_sum)))
