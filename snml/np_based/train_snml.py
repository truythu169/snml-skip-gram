from snml.np_based.model import Model
import time
import numpy as np
import argparse
import utils.tools as utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../../../output/snml/3000samples/31epochs/50dim/', type=str)
    parser.add_argument('--context_path', default='../../../data/text8/contexts/', type=str)
    parser.add_argument('--snml_train_file', default='../../../data/text8/scope.csv', type=str)
    parser.add_argument('--scope', default=100, type=int)
    args = parser.parse_args()

    # read snml train file
    utils.download_from_gcs(args.snml_train_file)
    data = np.genfromtxt(args.snml_train_file, delimiter=',').astype(int)

    # Run snml
    model = Model(args.model, args.context_path)
    snml_lengths = []
    print_step = 5
    start = time.time()
    for i in range(args.scope):
        w = data[i][0]
        c = data[i][1]

        length, probs = model.snml_length_sampling_multiprocess(w, c, epochs=31, neg_size=3000, n_context_sample=600)
        snml_lengths.append(length)

        # print process
        if (i + 1) % print_step == 0:
            end = time.time()
            print('Run {} step in: {:.4f} sec'.format(i + 1, (end - start)))
            start = time.time()

    print('{} scope snml length: {}'.format(args.scope, sum(snml_lengths)))

    # Save result to file
    filename = args.model + 'scope-{}-snml_length.txt'.format(args.scope)
    output = open(filename, 'w')
    for i in snml_lengths:
        output.write(str(i) + '\n')
    output.close()

    # upload to gcs
    utils.upload_to_gcs(filename, force_update=True)
