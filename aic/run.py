from aic.model import Model
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../../../output/text8/20200114/snml/1/train2/65dim/', type=str)
    parser.add_argument('--data', default='../../data/text8/train2.csv', type=str)
    args = parser.parse_args()

    model = Model(args.model, args.data)
    aic = model.aic()
    print('Final Log Likelihood: ', aic)
