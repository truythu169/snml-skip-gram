from collections import Counter
from utils.tools import load_pkl, save_pkl
import numpy as np


if __name__ == "__main__":
    # Data file
    raw_data_path = '../data/raw data/test.txt '
    context_to_dict_path = 'data/text8/dict/cont_to_int.dict'
    output_path = 'data/text8/contexts/distribution_from_raw.pkl'
    int_to_cont = load_pkl('data/text8/dict/int_to_cont.dict', local=True)

    # Load data
    with open(raw_data_path, encoding='utf-8') as f:
        words = f.read().split()

    # Load dict
    context_to_dict = load_pkl(context_to_dict_path, local=True)

    # Convert vocab to int
    context = []
    for word in words:
        if word in context_to_dict:
            context.append(context_to_dict[word])

    context_counts = Counter(context)
    n_context = len(context_to_dict)
    n_data = sum(list(context_counts.values()))

    context_distribution = np.zeros(n_context)
    for c, count in context_counts.items():
        context_distribution[c] = count / n_data

    context_distribution = np.array(context_distribution)
    save_pkl(context_distribution, output_path)
