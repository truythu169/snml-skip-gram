from utils.raw_dataset import RawDataset
import argparse
import csv
import numpy as np
import os
from utils.settings import config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../data/text8', type=str)
    parser.add_argument('--output', default='../data/processed data/', type=str)
    parser.add_argument('--batch_size', default=10000, type=int)
    parser.add_argument('--window_size', default=15, type=int)
    args = parser.parse_args()

    # make directories
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # initialize dataset
    data = RawDataset(args.input, args.output)

    # save processed data back to file
    print('Writing processed data back to file...')
    output = open(args.output + config['PREPROCESS']['output_data'], "w", newline='')
    writer = csv.writer(output)
    batches = data.get_batches(args.batch_size, args.window_size)
    count = 0
    for datum in batches:
        datum = np.array(datum).T
        np.random.shuffle(datum)
        writer.writerows(datum)
        count += datum.shape[0]
    output.close()

    print('{} rows of data processed!'.format(count))

