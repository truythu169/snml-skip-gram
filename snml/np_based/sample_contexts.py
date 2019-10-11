import os
import utils.tools as utils

if __name__ == "__main__":
    context_path = '../../../data/wiki/contexts/'
    n_context_sample = 600
    scope = 1000
    file_name = os.path.join(context_path, 'sample_contexts_{}.pkl'.format(n_context_sample))

    context_distribution = utils.load_pkl(context_path + 'context_distribution.pkl')
    if os.path.exists(file_name):
        print('Load file')
        contexts = utils.load_pkl(file_name)
    else:
        contexts = []

    print('Current contexts: ', len(contexts))

    # Sample contexts
    if scope + 1 > len(contexts):
        for i in range(scope - len(contexts)):
            samples = utils.sample_context(context_distribution, n_context_sample)
            contexts.append(samples)

    # Save result back to pkl
    utils.save_pkl(contexts, file_name)

    print(len(contexts))
    print(len(contexts[0][0]))
