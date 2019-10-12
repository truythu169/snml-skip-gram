from snml.np_based.model import Model
import time
import numpy as np
import argparse
import utils.tools as utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../../../output/text8/snml/3000samples/31epochs/50dim/', type=str)
    parser.add_argument('--context_path', default='../../../data/text8/contexts/', type=str)
    parser.add_argument('--snml_train_file', default='../../../data/text8/scope.csv', type=str)
    parser.add_argument('--scope', default=10, type=int)
    parser.add_argument('--epochs', default=31, type=int)
    args = parser.parse_args()

    # read snml train file
    utils.download_from_gcs(args.snml_train_file)
    data = np.genfromtxt(args.snml_train_file, delimiter=',').astype(int)

    # Run snml
    model = Model(args.model, args.context_path, n_context_sample=600)
    snml_lengths = []
    print_step = 2
    start = time.time()
    for i in range(10):
        w = data[0][0]
        c = data[0][1]

        length, probs = model.snml_length_sampling_multiprocess(w, c, epochs=args.epochs, neg_size=3000)
        snml_lengths.append(length)

        # print process
        if (i + 1) % print_step == 0:
            end = time.time()
            print('Run {} step in: {:.4f} sec'.format(i + 1, (end - start)))
            start = time.time()

    print('Mean: {:.4f} Min: {:.4f} Max: {:.4f} std: {:.4f}'.format(np.mean(snml_lengths), min(snml_lengths),
                                                                    max(snml_lengths), np.std(snml_lengths)))
