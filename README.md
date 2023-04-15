# Interpolating Neural Networks

This repo contains the source code for observing **double descent** phenomena with neural networks in **empirical asset pricing data**. 

![dd_curve](assets/new-bias-variance-risk-curve.png)
             Fig. A new double-U-shaped bias-variance risk curve for deep neural networks. (Image source: [original paper](https://arxiv.org/abs/1812.11118))

Deep learning models are heavily over-parameterized and can often get to perfect results on training data. In the traditional view, like bias-variance trade-offs, this could be a disaster that nothing may generalize to the unseen test data. However, as is often the case, such “overfitted” (training error = 0) deep learning models still present a decent performance on out-of-sample test data. 

As [Belkin et al.]() claimed that it is likely due to two reasons:
- The number of parameters is not a good measure of inductive bias, defined as the set of assumptions of a learning algorithm used to predict for unknown samples. See more discussion on DL model complexity in later sections.
- Equipped with a larger model, we might be able to discover larger function classes and further find interpolating functions that have smaller norm and are thus “simpler”.

## Usage
```
$ pip install interpolating-neural-networks
```

## Results



## Notes
- There are no regularization terms like weight decay, dropout.
- Each network is trained for a long time to achieve near-zero training risk. The learning rate is adjusted differently for models of different sizes.
- To make the model less sensitive to the initialization in the under-parameterization region, their experiments adopted a “weight reuse” scheme: the parameters obtained from training a smaller neural network are used as initialization for training larger networks.

## Citation

If you find this method and/or code useful, please consider citing

```
@misc{interpolatingneuralnetworks,
  author = {Akash Sonowal, Dr. Shankar Prawesh},
  title = {Interpolating Neural Networks in Asset Pricing},
  url = {https://github.com/akashsonowal/interpolating-neural-networks},
  year = {2022},
  note = "Version 0.0.1"
}
```
