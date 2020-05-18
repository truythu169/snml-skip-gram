from np_based.aic.model import AICModel
import utils.tools as utils
import numpy as np


class BICModel(AICModel):

    def __init__(self, model_path, data_file, context_path, n_negative_sample):
        super(BICModel, self).__init__(model_path, data_file, context_path, n_negative_sample)
        self.n_datums = utils.count_line(self.filename)

    def bic(self):
        return self.k * np.log(self.n_datums) - 2 * self.log_likelihood()
