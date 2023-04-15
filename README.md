# Interpolating Neural Networks

This repo contains the source code for observing double descent phenomena with neural networks in empirical asset pricing data.

What is Deep Double Descent?

Double descent phenomenon is when we increase the model size or number of training epochs, the performance first improves, then gets worse, and then improves again.

It has been shown in this paper by OpenAI (https://lnkd.in/dfpAeNbP) that the “double descent” phenomenon occurs in deep neural networks like CNNs, transformers etc.

According to the bias-variance trade-off in classical statistical learning theory, once the model complexity crosses a certain threshold, the model starts to overfit. It means if we increase the model complexity now, it will only increase the test error. Thus, after crossing a certain threshold, “large models are worse”.

This double descent challenges this conventional wisdom.

Key point: Because of the double descent, sufficiently large models undergo this behaviour where the test error first decreases then increases near the threshold (interpolation threshold - term used in the paper), and then decreases again.

## Usage
```
$ pip install interpolating-neural-networks
```

## Demo (Kaggle Notebook)

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
  note = "Version 1"
}
```
