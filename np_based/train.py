from np_based.model import SkipGram
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../../data/text8_ng4/', type=str)
    parser.add_argument('--output_path', default='../../output/skip_gram/text8_ng4/test/', type=str)
    parser.add_argument('--n_embedding', default=50, type=int)
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--n_sampled', default=15, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--print_step', default=1000000, type=int)
    parser.add_argument('--snml_dir', default='', type=str)
    parser.add_argument('--stop_threshold', default=0.000000001, type=float)
    args = parser.parse_args()
    snml = args.snml_dir != ''

    skip_gram = SkipGram(args.input_path, args.output_path, n_embedding=args.n_embedding,
                         n_sampled=args.n_sampled, epochs=args.epochs, batch_size=args.batch_size,
                         snml=snml, snml_dir=args.snml_dir)
    skip_gram.train(print_step=args.print_step, stop_threshold=args.stop_threshold, learning_rate=args.learning_rate)
    skip_gram.export_embedding()
    skip_gram.export_model()
