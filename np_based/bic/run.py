from np_based.bic.model import BICModel
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../../../output/sgns/text8/2/full/70dim/', type=str)
    parser.add_argument('--data_file', default='../../../data/text8/shufle/1/data.csv', type=str)
    parser.add_argument('--context_path', default='../../../data/text8/', type=str)
    parser.add_argument('--n_negative_sample', default=15, type=int)
    args = parser.parse_args()

    model = BICModel(args.model_path, args.data_file, args.context_path, args.n_negative_sample)
    bic = model.bic()
    aic = model.aic()
    print('Bayesian information criterion: ', bic)
    print('Akaike information criterion: ', aic)
