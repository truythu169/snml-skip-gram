import numpy as np


class Embedding:

    # Constructor
    def __init__(self, embedding_matrix, int_to_vocab_dict, vocab_to_int_dict):
        self.e = embedding_matrix
        self.w = int_to_vocab_dict
        self.i = vocab_to_int_dict
        self.n_vocab = embedding_matrix.shape[0]
        self.n_embed = embedding_matrix.shape[1]

    # Construct from file
    @staticmethod
    def from_file(filename):
        # load data from file
        f = open(filename, "r")
        words = []
        vectors = []
        for wn, line in enumerate(f):
            line = line.lower().strip().split()
            word = line[0]
            vector = [float(i) for i in line[1:]]
            words.append(word)
            vectors.append(vector)
        f.close()
        embedding_matrix = np.array(vectors)

        # create dict
        int_to_vocab = {ii: word for ii, word in enumerate(words)}
        vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

        # create new object
        embedding = Embedding(embedding_matrix, int_to_vocab, vocab_to_int)

        return embedding

    def word(self, index):
        return self.w[index]

    def words(self, indexes):
        return [self.w[i] for i in indexes]

    def index(self, word):
        return self.i[word]

    def indexes(self, words):
        return [self.i[i] for i in words]

    def vector(self, index):
        # convert vocab to int
        if isinstance(index, str):
            index = self.index(index)

        return self.e[index]

    def vectors(self, indexes):
        # convert list of vocab to int
        if isinstance(indexes[0], str):
            indexes = self.indexes(indexes)

        return self.e[indexes]

    def in_vocab(self, word):
        return word in self.i

    def in_vocabs(self, words):
        for word in words:
            if not word in self.i:
                return False
        return True

    # Save to file
    def save(self, filename):
        output = open(filename, 'w')
        for i in range(self.n_vocab):
            line = self.w[i] + ' ' + ' '.join([str(j) for j in self.e[i]]) + '\n'
            output.write(line)

        output.close()