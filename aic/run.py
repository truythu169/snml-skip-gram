from aic.model import Model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../../output/text8/momentum/snml/1/50dim/', type=str)
    parser.add_argument('--data', default='../../data/text8/', type=str)
    args = parser.parse_args()

    model = Model(args.model, args.data)
    aic = model.aic()
    print('Final Log Likelihood: ', aic)
