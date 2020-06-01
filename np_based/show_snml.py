import numpy as np
import pickle
import argparse


def load_txt(filename):
    """ Load data to pickle """
    # Load txt
    data = []
    fp = open(filename, 'r')
    line = fp.readline()
    while line:
        line = fp.readline()
        line = line.strip()
        try:
            data.append(float(line))
        except:
            continue
    #             print(line)
    fp.close()

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default='../output/sgns/text8/2/train1/{}dim/scope-2692279-snml_length.txt', type=str)
    args = parser.parse_args()

    dim_list = [40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130, 140, 150, 160, 170, 200]
    snml_list = []

    for dim in dim_list:
        snml_list.append(np.array(load_txt(args.file_path.format(dim))))

    snml_length = len(snml_list[0])
    for i in range(len(dim_list)):
        print(sum(snml_list[i]) / snml_length)
