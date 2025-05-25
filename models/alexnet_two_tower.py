# PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. python3 models/alexnet_two_tower.py

# Paper: https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
# Implementation: https://github.com/akrizhevsky/cuda-convnet2

# Introduction ***
# subsets of ImageNet used in the ILSVRC-2010 and ILSVRC-2012
# 1.2 million labeled training examples
# network contains five convolutional and three fully-connected layers
# network takes between five and six days to train on two GTX 580 3GB GPUs

# The Dataset ***
# ImageNet is a dataset of over 15 million labeled high-resolution images belonging to roughly 22,000 categories
# ILSVRC uses a subset of ImageNet with roughly 1000 images in each of
# 1000 categories. In all, there are roughly 1.2 million training images, 50,000 validation images, and 150,000 testing images
# two error rates:
# top-1 and top-5, where the top-5 error rate is the fraction of test images for which the correct label
# is not among the five labels considered most probable by the model
# we down-sampled the images to a fixed resolution of 256 × 256.
# Given a rectangular image, we first rescaled the image such that the shorter side was of length 256, and then
# cropped out the central 256×256 patch from the resulting image
# We did not pre-process the images in any other way, except for subtracting the mean activity over the training set
# from each pixel. So we trained our network on the (centered) raw RGB values of the pixels.

from typing import Optional
from dataclasses import dataclass
from functools import partial
from tqdm import trange

import torch
from torch import Tensor
from torch import nn


def init_module(m: nn.Module, val: float=0.):
  if isinstance(m, (nn.Linear, nn.Conv2d)):
    nn.init.normal_(m.weight, mean=0, std=1e-2)
    nn.init.constant_(m.bias, val=val)


def convolution(in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0,
                maxpool: bool=True, init_bias: Optional[float]=None) -> nn.Sequential:
  layers = [
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
    nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
    nn.ReLU(inplace=True),
  ]
  if maxpool: layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
  init_fn = init_module if init_bias is None else partial(init_module, val=init_bias)
  return nn.Sequential(*layers).apply(init_fn)

def fully_connected(in_features: int, out_features: int, final_layer: bool=False,
                    init_bias: Optional[float]=None, dropout: float=0.) -> nn.Sequential:
  layers = []
  if not final_layer: layers.append(nn.Dropout(dropout))
  layers.append(nn.Linear(in_features, out_features))
  if not final_layer: layers.append(nn.ReLU(inplace=True))
  init_fn = init_module if init_bias is None else partial(init_module, val=init_bias)
  return nn.Sequential(*layers).apply(init_fn)


class AlexNet(nn.Module):
  def __init__(self, num_classes: int=1000, dropout: float=0.5):
    super().__init__()
    # TODO: mention how the layers split into two towers are represented here
    self.features = nn.Sequential(
      convolution(3, 96, 11, stride=4),
      convolution(96, 256, 3, padding=2, init_bias=1.),
      convolution(256, 384, 3, padding=1, maxpool=False),
      convolution(384, 384, 3, padding=1, maxpool=False, init_bias=1.),
      convolution(384, 256, 3, padding=1, init_bias=1.),
    )
    self.classifier = nn.Sequential(
      fully_connected(9216, 4096, init_bias=1., dropout=dropout),
      fully_connected(4096, 4096, init_bias=1., dropout=dropout),
      fully_connected(4096, num_classes, final_layer=True),
    )
    print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))  # should be roughly 60 million

  def forward(self, x: Tensor):
    x = self.features(x)
    x = x.flatten(1)
    x = self.classifier(x)
    return x

  def get_num_params(self):
    n_params = sum(p.numel() for p in self.parameters())
    return n_params


@dataclass
class TrainConfig:
  epochs: int = 90  # roughy 90 cycles
  batch_size: int = 128
  # TODO: https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  lr: float = 1e-2  # manually adjusted 3 times as validation loss saturated
  decay: float = 5e-4
  momentum: float = 0.9


def train(X_train: Tensor, Y_train: Tensor, num_classes: int, config: TrainConfig=None, device='cpu'):
  config = TrainConfig() if config is None else config
  X_train, Y_train = X_train.to(device), Y_train.to(device)
  model = AlexNet(num_classes).to(device)
  model.train()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(
    model.parameters(), lr=config.lr, weight_decay=config.decay, momentum=config.momentum
  )
  num_batches = X_train.size(0) // config.batch_size
  for epoch in (t:=trange(config.epochs)):
    for batch in range(num_batches):
      start, stop = batch * config.batch_size, (batch + 1) * config.batch_size
      out = model(X_train[start:stop])
      loss: Tensor = criterion(out, Y_train[start:stop])

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      progress = f'Epoch [{epoch+1:3d}/{config.epochs:3d}], Step [{batch+1:3d}/{num_batches:3d}], Loss: {loss.item():.4f}'
      t.set_description(progress)
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

  training_config = TrainConfig(epochs=15, batch_size=8)
  train(sample_X, sample_Y, num_classes, training_config, device.type)
