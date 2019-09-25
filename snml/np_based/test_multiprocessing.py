from snml.np_based.model import Model
from sklearn.metrics import mean_absolute_error
import time


if __name__ == "__main__":

    model = Model('../models/100dim/',
                  '../context_distribution.pkl')

    start = time.time()
    snml_length, probs1 = model.snml_length_sampling_multiprocess(8229, 9023, epochs=31, neg_size=3000,
                                                                  n_context_sample=600)
    end = time.time()
    print("Multiprocessing in {:.4f} sec".format(end - start))
    print(snml_length)

    start = time.time()
    snml_length, probs2 = model.snml_length_sampling(8229, 9023, epochs=31, neg_size=3000, n_context_sample=600)
    end = time.time()
    print("Single process in {:.4f} sec".format(end - start))
    print(snml_length)

    # for i in range(len(probs1)):
    #     print(probs1[i], probs2[i])
    print(mean_absolute_error(probs1, probs2))



