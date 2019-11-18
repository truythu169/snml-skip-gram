from bic.model import BICModel
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../../output/text8/momentum/snml/1/50dim/', type=str)
    parser.add_argument('--data', default='../../data/text8/', type=str)
    args = parser.parse_args()

    model = BICModel(args.model, args.data)
    bic = model.bic()
    aic = model.aic()
    print('Bayesian information criterion: ', bic)
    print('Akaike information criterion: ', aic)
