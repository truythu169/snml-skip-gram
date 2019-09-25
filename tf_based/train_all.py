from tf_based.skip_gram import SkipGram

if __name__ == "__main__":
    input_path = '../data/processed data/'
    output_path = '../output/1/'

    dimension_list = [25, 75, 125, 175, 225, 275]
    # dimension_list = [50, 100, 150, 200, 250, 300]

    for dimension in dimension_list:
        skip_gram = SkipGram(input_path, output_path, n_embedding=dimension)
        skip_gram.train(n_sampled=200, epochs=20, batch_size=10000, print_step=1000)
        skip_gram.export_embedding()
