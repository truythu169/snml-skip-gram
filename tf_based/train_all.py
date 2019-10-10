from tf_based.skip_gram import SkipGram

if __name__ == "__main__":
    input_path = '../data/text8/'
    output_path = '../output/1/'

    dimension_list = [25, 75, 125, 175, 225, 275]

    for dimension in dimension_list:
        skip_gram = SkipGram(input_path, output_path, n_embedding=dimension)
        skip_gram.train(n_sampled=3000, epochs=31, batch_size=10000, print_step=1000)
        skip_gram.export_embedding()
