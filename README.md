# NNGP: Deep Neural Network Kernel for Gaussian Process

TensorFlow open source implementation of

[**Deep Neural Networks as Gaussian Processes**](https://arxiv.org/abs/1711.00165)


by Jaehoon Lee*, Yasaman Bahri*, Roman Novak, Sam Schoenholz, Jeffrey Pennington,
Jascha Sohl-dickstein

Presented at the International Conference on Learning Representation(ICLR) 2018.

## UPDATE (September 2020):
See also [Neural Tangents: Fast and Easy Infinite Neural Networks in Python](https://arxiv.org/abs/1912.02803) (ICLR 2020)
available at [github.com/google/neural-tangents](https://github.com/google/neural-tangents) for 
more up-to-date progress on computing NNGP as well as NT kernels supporting wide variety of architectural components.


## Overview
A deep neural network with i.i.d. priors over its parameters is equivalent to a 
Gaussian process in the limit of infinite network width. The Neural Network
Gaussian Process (NNGP) is fully described by a covariance kernel determined by 
corresponding architecture.

This code constructs covariance kernel for the Gaussian process that is equivalent to
infinitely wide, fully connected, deep neural networks. 

## Usage

To use the code, run `run_experiments.py`,
which uses NNGP kernel to make full Bayesian prediction on the MNIST dataset.


```python
python run_experiments.py \
       --num_train=100 \
       --num_eval=10000 \
       --hparams='nonlinearity=relu,depth=100,weight_var=1.79,bias_var=0.83' \
```

## Contact
***Code author:*** Jaehoon Lee, Yasaman Bahri, Roman Novak

***Pull requests and issues:*** @jaehlee

## Citation
If you use this code, please cite our paper:
```
  @article{
    lee2018deep,
    title={Deep Neural Networks as Gaussian Processes},
    author={Jaehoon Lee, Yasaman Bahri, Roman Novak, Sam Schoenholz, Jeffrey Pennington, Jascha Sohl-dickstein},
    journal={International Conference on Learning Representations},
    year={2018},
    url={https://openreview.net/forum?id=B1EA-M-0Z},
  }
```

## Note

This is not an official Google product.
