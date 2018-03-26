import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import eager as tfe

sys.path.append('..')
from universalgp import inf, cov, lik
from datasets import sensitive_example

CHECKPOINT_PATH = "/its/home/tk324/tensorflow/m1/ckpt-504"
INPUT_DIM = 2
OUTPUT_DIM = 1
NUM_TRAIN = 300
NUM_INDUCING = 300
NUM_COMPONENTS = 1
NUM_SAMPLES = 1000
DIAG_POST = False
LIKELIHODD = lik.LikelihoodLogistic
RESULT_PATH = "./predictions.npz"

tfe.enable_eager_execution()

def main():
    """This scripts makes predictions from a trained model

    First train the model like this:

    ```
    python gaussian_process.py --data=sensitive_example --save_dir=/its/home/tk324/tensorflow --model_name=m1
    ```

    You can of course also choose a different directory and model name.
    """
    # load GP model from checkpoint
    with tfe.restore_variables_on_create(CHECKPOINT_PATH):
        gp = inf.Variational([cov.SquaredExponential(INPUT_DIM) for _ in range(OUTPUT_DIM)],
                             LIKELIHODD(),
                             NUM_TRAIN,
                             NUM_INDUCING,
                             {'num_components': NUM_COMPONENTS, 'num_samples': NUM_SAMPLES, 'diag_post': DIAG_POST})

    # pred_means, pred_vars = gp.predict([[1., 1.]])
    # print(pred_means)

    # make predictions
    dataset = sensitive_example()
    pred_means, pred_vars = gp.predict(dataset.xtest)
    # save in file
    np.savez_compressed(RESULT_PATH, **{'pred_means': pred_means, 'pred_vars': pred_vars})

    ##### To load the file again:
    # loaded_data = np.load("predictions.npz")
    # pred_means = loaded_data['pred_means']

if __name__ == "__main__":
    main()
