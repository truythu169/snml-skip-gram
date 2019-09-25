import utils.tools as utils
import time


if __name__ == "__main__":
    start_time = time.time()
    samples = utils.sample_learning_data('../data/processed data/split', 12803, 1000)
    end_time = time.time()
    print(samples)
    print("--- %s seconds ---" % (end_time - start_time))

    '''
    # initialize model
    model = Model()
    model.load_model('snml/models/100dim')
    print('Model loaded:')
    print('Vocabulary size: {} words'.format(model.V))
    print('Embedding size: {} dimensions'.format(model.K))
    print('Context size: {} words'.format(model.V_dash))

    # read data
    # data = np.genfromtxt('data/processed data/test.csv', delimiter=',').astype(int)

    for i in range(10):
        # neg = utils.sample_negative(neg_size=100, except_sample={369}, vocab_size=model.V_dash)
        # loss, prob = model.train_neg(1637, 369, neg, True)
        loss, prob = model.train(1637, 369)
        print(loss, prob)
    '''