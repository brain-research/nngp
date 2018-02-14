# NNGP: Deep Neural Network Kernel for Gaussian Process

TensorFlow open source implementation of 

[*Deep Neural Networks as Gaussian Processes*](https://arxiv.org/abs/1711.00165)

by Jaehoon Lee, Yasaman Bahri, Roman Novak, Sam Schoenholz, Jeffrey Pennington,
Jascha Sohl-dickstein (To appear in ICLR 2018)

--
The code constructs covariance kernel for Gaussian Process that is equivalent to
infinitely wide, fully connected, deep neural networks. `run_experiments.py`
uses this kernel to make full Bayesian prediction on MNIST dataset.

--
Usage:

```python
python run_experiments.py \
       --num_train=100 \
       --num_eval=1000 \
       --hparams='nonlinearity=relu,depth=10,weight_var=2.0,bias_var=0.2' \
       --max_gauss=10
```

## Contact
***code author:*** Jaehoon Lee, Yasaman Bahri

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
