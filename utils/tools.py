import numpy as np
from collections import Counter
from google.cloud import storage as gcs
import random
import pickle
import os
from nltk.corpus import stopwords
from utils.settings import config, env


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = env['GCS']['app_credential']
project_id = env['GCS']['project_id']
bucket_name = env['GCS']['bucket']


def preprocess(text, min_word=20):
    # Load stop words
    stop_words = stopwords.words('english')
    n_top = int(config['PREPROCESS']['n_top'])

    # Replace punctuation with tokens so we can use them in our model
    # text = text.lower()
    # text = text.replace('.', ' <PERIOD> ')
    # text = text.replace(',', ' <COMMA> ')
    # text = text.replace('"', ' <QUOTATION_MARK> ')
    # text = text.replace(';', ' <SEMICOLON> ')
    # text = text.replace('!', ' <EXCLAMATION_MARK> ')
    # text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('(', ' <LEFT_PAREN> ')
    # text = text.replace(')', ' <RIGHT_PAREN> ')
    # text = text.replace('--', ' <HYPHENS> ')
    # text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    # text = text.replace(':', ' <COLON> ')
    words = text.split()

    # Remove all words with  min_word or fewer occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > min_word]

    # Get top words
    clear_words = [word for word in words if (word.lower() not in stop_words) and (len(word) > 1)]
    word_counts = Counter(clear_words)
    top_words = word_counts.most_common(n_top)

    return trimmed_words, top_words


def get_train_words(int_words):
    """ implementation of subsampling """
    word_counts = Counter(int_words)
    total_count = len(int_words)
    threshold = float(config['PREPROCESS']['threshold'])
    freqs = {word: count / total_count for word, count in word_counts.items()}
    p_drop = {word: 1 - np.sqrt(threshold / freqs[word]) for word in word_counts}
    train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

    return train_words


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict....
    """
    # Load stop words
    stop_words = stopwords.words('english')

    # dict for contexts
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    int_to_cont = {ii: word for ii, word in enumerate(sorted_vocab)}
    cont_to_int = {word: ii for ii, word in int_to_cont.items()}

    # dict for words
    non_stopwords_words = [word for word in words if (word.lower() not in stop_words) and (len(word) > 1)]
    word_counts = Counter(non_stopwords_words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return [vocab_to_int, int_to_vocab, cont_to_int, int_to_cont]


def save_pkl(data, filename, local=False):
    """ Save data to file """
    output = open(filename, 'wb')
    pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)
    output.close()

    # upload to gcs
    if not local:
        upload_to_gcs(filename, force_update=True)


def load_pkl(filename, local=False):
    """ Load data to pickle """
    # download from gcs
    if not local:
        download_from_gcs(filename, force_update=True)

    input = open(filename, 'rb')
    data = pickle.load(input)
    input.close()
    return data


def convert_local_path_to_gcs(local_file):
    parts = local_file.split('.')
    gcs_file = parts[-2][1:] + '.' + parts[-1]
    return gcs_file


def download_from_gcs(local_path, force_update=False):
    if env['GCS']['sync'] == 'no':
        return

    gcs_path = convert_local_path_to_gcs(local_path)
    client = gcs.Client(project_id)
    bucket = client.get_bucket(bucket_name)
    blob = gcs.Blob(gcs_path, bucket)

    if os.path.exists(local_path) and not force_update:
        return

    output_dictionary = os.path.dirname(local_path)
    if not os.path.exists(output_dictionary):
        os.makedirs(output_dictionary)

    blob.download_to_filename(local_path)


def upload_to_gcs(local_path, force_update=False):
    if env['GCS']['sync'] == 'no':
        return

    gcs_path = convert_local_path_to_gcs(local_path)
    client = gcs.Client(project_id)
    bucket = client.get_bucket(bucket_name)
    blob = gcs.Blob(gcs_path, bucket)

    if blob.exists() and not force_update:
        return

    blob.upload_from_filename(local_path)


def label_binarizer(labels, n_class):
    """ Convert dense labels array to sparse labels matrix """
    n_records = len(labels)
    labels_b = np.zeros((n_records, n_class))
    labels_b[np.arange(n_records), labels] = 1

    return labels_b


def sample_negative(neg_size=200, except_sample=None, vocab_size=200):
    if not except_sample:
        except_sample = {}

    negative_samples = []
    while len(negative_samples) != neg_size:
        # random a sample
        sample = random.randint(0, vocab_size - 1)
        if sample not in except_sample:
            negative_samples.append(sample)

    return negative_samples


def sample_contexts(context_distribution, context_path, sample_size=1000, loop=0, from_file=True):
    if not from_file:
        samples = _sample_context(context_distribution, sample_size)
        return samples

    # Check if sample context file exits
    file_name = os.path.join(context_path, 'sample_contexts_{}.pkl'.format(sample_size))
    if os.path.exists(file_name):
        contexts = load_pkl(file_name)
    else:
        contexts = []

    # Sample contexts
    if loop + 1 > len(contexts):
        for i in range(loop + 1 - len(contexts)):
            samples = _sample_context(context_distribution, sample_size)
            contexts.append(samples)

        # Save result back to pkl
        save_pkl(contexts, file_name)

    return contexts[loop]


def _sample_context(context_distribution, sample_size=1000):
    draw = np.random.multinomial(sample_size, context_distribution)
    sample_ids = np.where(draw > 0)[0]

    samples = []
    samples_prob = []
    for context_id in sample_ids:
        samples.extend([context_id] * draw[context_id])
        samples_prob.extend([context_distribution[context_id]] * draw[context_id])

    return samples, samples_prob


def sample_learning_data(data_path, max_n_file, rand_size):
    file_no = random.randint(0, max_n_file)
    file_name = data_path + ('x%05d' % file_no)
    data = []

    # Read data file
    with open(file_name) as f:
        line = f.readline()
        while line:
            line = line.split(',')
            word = int(line[0])
            context = int(line[1])
            data.append([word, context])
            line = f.readline()

    # select rand_size random records
    data = np.array(data)
    n_record = len(data)
    ids = random.sample(range(n_record), rand_size)
    data = data[ids]

    words = data[:, 0].tolist()
    contexts = data[:, 1].tolist()

    return words, contexts


