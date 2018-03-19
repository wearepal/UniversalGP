# UniversalGP

UniversalGP: A basic generic implementation for Gaussian process models

The code was tested on Python 3.6 and Tensorflow 1.5.

Based on the code [AutoGP: Automated Variational Inference for Gaussian
Process Models](https://github.com/ebonilla/AutoGP), UniversalGP has 
many developments and improvements.

1. Specify the inference method in the model setting, including 
   variational inference, leave-one-out inference and exact inference.

   * Variational inference can be used for generic Gaussian process 
     models( multi-input, multi-output, regression and classification) 
     with black-box likelihood.

   * Leave-one-out inference can be used for generic Gaussian process 
     models( multi-input, multi-output, regression and classification) 
     with Gaussian likelihood.

   * Exact inference can be only used for standard Gaussian process 
     model (one dimensional output) with Gaussian likelihood.

2. Improve and adjust some codes to Tensorflow running more smoothly

## Usage

### Train with included dataset

To train the GP with one of the included datasets, run for example
```sh
python train_gp.py --data=mnist
```
This will train on the MNIST dataset. For other included datasets see 
the files in `datasets/`.

By default, the code uses *Variational Inference*. In order to use a 
different method, specify the `--inf` parameter. For example
```sh
python train_gp.py --data=mnist --inf=Exact
```
uses *Exact Inference* (which can only be used for regression tasks).  
Other allowed values for `--inf` are `Variational` and `Loo`.

Other useful parameters:

* `--lik`: the likelihood; allowed values are `LikelihoodGaussian` 
  (default) and `LikelihoodSoftmax`
* `--metric`: which metric to use for evaluation of the model; allowed 
  values are `rmse` (default) and `accuracy`
* `--train_steps`: the number of training steps (default: 500)

To see all available parameters with explanations and default values, 
run
```sh
python train_gp.py --helpfull
```

### Add a new dataset

To add a new dataset, create a new python file in `datasets/`. In this 
file, create a function which returns a dataset according to the 
definition in `datasets/definition.py`. The resulting file should look 
something like this:
```python
from .definition import Dataset

def my_dataset():

    # ...
    # read data from files
    # ...

    return Dataset(train_fn=lambda: train_set,
                   test_fn=lambda: test_set,
                   num_train=100,
                   input_dim=28,
                   inducing_inputs=inducing_inputs_numpy,
                   output_dim=1)
```

The last step is to import the new function in `datasets/__init__.py`.  
If you named the file `my_dataset` and the function also `my_dataset()` 
then add the following line to `datasets/__init__.py`:
```python
from .my_dataset import my_dataset
```

Now you can use this data by using the parameter `--data=my_dataset`.

### Call a trained model from your code

When you are done with training, you might want to used the trained 
model for something. Here is how to do that.

First, we need to make sure the trained model is saved. This is done by 
specifying the `--save_dir` parameter. This parameter expects the path 
to a directory where the model can be saved. UniversalGP will create a 
new directory in this directory with the name given by `--model_name` 
(default name is "local"). If there is already a saved model in there, 
UniversalGP will use this as the initial values and continue training 
from there.

When training with these parameters, UniversalGP creates checkpoints in 
`save_dir/model_name`. The frequency of checkpoint creation is 
determined by `--chkpnt_steps`. The name of the checkpoint file includes 
the training step at which it was created. For example `ckpt-500` for a 
checkpoint created at step 500.

To use this trained model do:
```python
import tensorflow.contrib.eager as tfe  # eager execution
from universalgp import inf, cov, lik

# restore all variables from checkpoint
with tfe.restore_variables_on_create("save_dir/model_name/chkpt-500"):
    gp = inf.Variational(cov.SquaredExponential(input_dim),
                         lik.LikelihoodGaussian(),
                         num_train=3756,
                         num_inducing=100)

prediction = gp.predict(test_inputs=[[3.2], [4.6]])
print(prediction)
print(gp.get_all_variables())  # print the values of all train variables
```
