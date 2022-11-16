# Discrete Latent Space (In progress......)


## Installing Dependencies

To install dependencies, create a conda or virtual environment with Python 3 and then run `pip install -r requirements.txt`. 

## Running  

Simply run `python3 main.py`. Make sure to include the `-save` flag if you want to save your model. You can also add parameters in the command line. The default values are specified below:

```python
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
```

## Models

The DLS has the following fundamental model components:

1. An `Encoder` class which defines the map `x -> z_e`
2. A `VectorQuantizer` class which transform the encoder output into a discrete one-hot vector that is the index of the closest embedding vector `z_e -> z_q`
3. A `Decoder` class which defines the map `z_q -> x_hat` and reconstructs the original image

The Encoder / Decoder classes are convolutional and inverse convolutional stacks, which include Residual blocks in their architecture [see ResNet paper](https://arxiv.org/abs/1512.03385). The residual models are defined by the `ResidualLayer` and `ResidualStack` classes.

