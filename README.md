# interpolating-neural-networks
This repo contains the source code for observing double descent phenomena with neural networks in empirical asset pricing data.

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
