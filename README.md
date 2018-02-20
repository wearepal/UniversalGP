# UniversalGP
UniversalGP: A basic generic implementation for Gaussian process models

The code was tested on Python3.6 and TensorFlow1.5

Based on the code [AutoGP: Automated Variational Inference for Gaussian Process Models](
https://github.com/ebonilla/AutoGP), UniversalGP has many developments and improvements.

1. Specify the inference method in the model setting, including variational inference, leave-one-out inference and exact inference. 

Variational inference can be used for generic Gaussian process models( multi-input, multi-ouput, regression and classification) with black-box likelihood.

Leave-one-out inference can be used for generic Gaussian process models( multi-input, multi-ouput, regression and classification) with Gaussian likelihood

Exact inference can be only used for standard Gaussian process model (one dimensional output) with Gaussian likelihood

2. Improve and adjust some codes to TensorFlow running more smoothly



 




