from snml.np_based.model_momentum import ModelMomentum
from multiprocessing import Pool
import multiprocessing
import utils.tools as utils


if __name__ == "__main__":
    words = [6581, 93, 4519, 506]
    contexts = [390, 1172, 1545, 22]

    model = ModelMomentum('../../../output/text8/snml/3000samples/31epochs/60dim/',
                          '../../../data/text8/contexts/', n_context_sample=600)

    for i in range(len(words)):
        word = words[i]
        context = contexts[i]

        # Update all other context
        print('Start: ', word)

        # implement pools
        job_args = [(word, c, 31, 3000) for c in range(model.V_dash)]
        p = Pool(multiprocessing.cpu_count())
        probs = p.map(model._train_job, job_args)
        p.close()
        p.join()

        # save context's probs
        utils.save_pkl(probs, '../../../output/test/contexts_probs_{}.pkl'.format(word))
