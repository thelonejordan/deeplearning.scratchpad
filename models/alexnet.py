# PYTORCH_ENABLE_MPS_FALLBACK=1 python3 models/alexnet.py

# Paper: https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
# Implementation: https://github.com/akrizhevsky/cuda-convnet2
# Paper: https://arxiv.org/abs/1404.5997

# Introduction ***
# subsets of ImageNet used in the ILSVRC-2010 and ILSVRC-2012
# 1.2 million labeled training examples
# network contains five convolutional and three fully-connected layers
# network takes between five and six days to train on two GTX 580 3GB GPUs

# The Dataset ***
# ImageNet is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories
# ILSVRC uses a subset of ImageNet with roughly 1000 images in each of
# 1000 categories. In all, there are roughly 1.2 million training images, 50,000 validation images, and
# 150,000 testing images
# two error rates:
# top-1 and top-5, where the top-5 error rate is the fraction of test images for which the correct label
# is not among the five labels considered most probable by the model
# we down-sampled the images to a fixed resolution of 256 × 256.
# Given a rectangular image, we first rescaled the image such that the shorter side was of length 256, and then
# cropped out the central 256×256 patch from the resulting image
# We did not pre-process the images in any other way, except for subtracting the mean activity over the training set
# from each pixel. So we trained our network on the (centered) raw RGB values of the pixels.

from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor
from torch import nn

def init_weights(m: nn.Module):
  if isinstance(m, (nn.Linear, nn.Conv2d)):
    nn.init.normal_(m.weight, mean=0, std=1e-2)

def init_biases(m: nn.Module, val: float=0.):
  if isinstance(m, (nn.Linear, nn.Conv2d)):
    nn.init.constant_(m.bias, val=val)

class AlexNet(nn.Module):
  def __init__(self, num_classes: int=1000):
    super().__init__()

    # these layers have no learnable parameters
    activation = nn.ReLU()
    maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
    norm = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)

    self.model = nn.Sequential(
      # convolutional layer 1
      nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4).apply(init_biases),
      norm, activation, maxpool,
      # convolutional layer 2
      nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, padding=2).apply(partial(init_biases, val=1.)),
      norm, activation, maxpool,
      # convolutional layer 3
      nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1).apply(init_biases),
      norm, activation,
      # convolutional layer 4
      nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1).apply(partial(init_biases, val=1.)),
      norm, activation,
      # convolutional layer 5
      nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1).apply(partial(init_biases, val=1.)),
      norm, activation, maxpool,
      # fully connected layer 1
      nn.Flatten(start_dim=1), nn.Dropout(0.5), nn.Linear(9216, 4096).apply(partial(init_biases, val=1.)),
      activation,
      # fully connected layer 2
      nn.Dropout(0.5), nn.Linear(4096, 4096).apply(partial(init_biases, val=1.)),
      activation,
      # fully connected layer 3
      nn.Linear(4096, num_classes).apply(init_weights).apply(init_biases)
    ).apply(init_weights)

  def forward(self, x: Tensor):
    return self.model(x)

@dataclass
class TrainConfig:
  epochs: int = 90 # roughy 90 cycles
  batch_size: int = 128
  lr: float = 1e-2 # manually adjusted 3 times as validation loss saturated
  decay: float = 5e-4
  momentum: float = 0.9

def train(X_train: Tensor, Y_train: Tensor, num_classes: int, config: TrainConfig=None, device='cpu'):
  config = TrainConfig() if config is None else config
  X_train, Y_train = X_train.to(device), Y_train.to(device)
  model = AlexNet(num_classes).to(device)
  count_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
  print(f"Parameter Count: {count_params/1e6:.2f} M" ) # should be 60 million
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(
    model.parameters(), lr=config.lr, weight_decay=config.decay, momentum=config.momentum)

  num_batches = X_train.size(0) // config.batch_size
  for epoch in range(config.epochs):
    for batch in range(num_batches):
      start, stop = batch * config.batch_size, (batch + 1) * config.batch_size
      out = model(X_train[start:stop])
      loss: Tensor = criterion(out, Y_train[start:stop])

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      print(f'Epoch [{epoch+1:3d}/{config.epochs:3d}], Step [{batch+1:3d}/{num_batches:3d}], Loss: {loss.item():.4f}')

  return model

if __name__ == "__main__":
  from helpers import set_device, set_seed
  device = set_device()
  set_seed(device)

  num_classes = 1000
  N, C, H, W = 32, 3, 224, 224
  sample_X = torch.randn(N, C, H, W)
  sample_Y = torch.randint(0, num_classes, (N,))
  # print(sample_X.shape, sample_Y.shape)

  training_config = TrainConfig(epochs=10, batch_size=8)
  train(sample_X, sample_Y, num_classes, training_config, device)
