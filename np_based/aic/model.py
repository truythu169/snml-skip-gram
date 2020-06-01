from np_based.snml_model import Model
import time
import numpy as np
import csv


class AICModel:

    # Constructor
    def __init__(self, model_path, data_file, context_path, n_negative_sample):
        # SNML model
        self.snml_model = Model(model_path, context_path, n_negative_sample)

        # Hyper parameters
        self.k = self.snml_model.E.shape[0] * self.snml_model.E.shape[1] + \
                 self.snml_model.F.shape[0] * self.snml_model.F.shape[1]
        self.sum_log_likelihood = 0

        # Display shapes
        print('Matrix E: Vw * d: ', self.snml_model.E.shape[0], self.snml_model.E.shape[1])
        print('Matrix F: d * Bc: ', self.snml_model.F.shape[0], self.snml_model.F.shape[1])
        print('k: ', self.k)

        # paths
        self.model_path = model_path
        self.filename = data_file

    def log_likelihood(self, print_step=1000000):
        if self.sum_log_likelihood == 0:
            sum_log_likelihood = 0
            iteration = 0

            start = time.time()
            with open(self.filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Print step
                    iteration += 1
                    if iteration % print_step == 0:
                        end = time.time()
                        print("Iteration: {}".format(iteration),
                              "Sum log likelihood: {:.4f}".format(np.mean(sum_log_likelihood)),
                              "{:.4f} sec/ {} sample".format((end - start), print_step))

                        start = time.time()

                    sum_log_likelihood -= self.snml_model.validation_loss(int(row[0]), int(row[1]))

            self.sum_log_likelihood = sum_log_likelihood

        return self.sum_log_likelihood

    def aic(self):
        return 2 * self.k - 2 * self.log_likelihood()
