# python3 models/resnet.py

# https://arxiv.org/abs/1512.03385
# https://d2l.ai/chapter_convolutional-modern/resnet.html
# https://github.com/tinygrad/tinygrad/blob/master/extra/models/resnet.py
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnetv2.py

import os
import torch
from torch import nn, Tensor
from helpers import timeit

class ResidualBlock(nn.Module):
  def __init__(self, filters: int, size: int=3, subsample: bool=False):
    super().__init__()
    self.stride = 2 if subsample else 1
    in_filters = filters // self.stride
    self.c1 = nn.Conv2d(in_filters, filters, kernel_size=size, stride=self.stride, padding=1, bias=False)
    self.bn1   = nn.BatchNorm2d(filters)
    self.c2 = nn.Conv2d(filters, filters, size, padding=1, bias=False)
    self.bn2   = nn.BatchNorm2d(filters)
    self.act = nn.ReLU(inplace=True)
    self.proj = nn.Conv2d(in_filters, filters, 1, stride=2, bias=False) if subsample else None

  def forward(self, x: Tensor):
    identity = x
    z = self.act(self.bn1(self.c1(x)))
    z = self.bn2(self.c2(z))
    identity = self.proj(x) if self.proj else identity
    return self.act(z)

class Resnet(nn.Module):
  def __init__(self, layers: int=18, num_classes: int=1000):
    super().__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.res = self.build(layers)
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(512, num_classes)
    self.sm = nn.Softmax(dim=-1)
    self.initialize()

  def initialize(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)

  def forward(self, x: Tensor):
    x = self.maxpool(self.conv1(x))
    x = self.res(x)
    x = self.avgpool(x)
    x = self.fc(x.flatten(1))
    x = self.sm(x)
    return x

  @staticmethod
  def build(layers: int=18) -> nn.Module:
    config = {
      "18": [
        {"blocks": 2, "layers": [{ "size": 3, "filters": 64},  { "size": 3, "filters": 64}]},
        {"blocks": 2, "layers": [{ "size": 3, "filters": 128}, { "size": 3, "filters": 128}]},
        {"blocks": 2, "layers": [{ "size": 3, "filters": 256}, { "size": 3, "filters": 256}]},
        {"blocks": 2, "layers": [{ "size": 3, "filters": 512}, { "size": 3, "filters": 512}]}
      ],
      "34": [
        { "blocks": 3, "layers": [{ "size": 3, "filters": 64},  { "size": 3, "filters": 64}]},
        { "blocks": 4, "layers": [{ "size": 3, "filters": 128}, { "size": 3, "filters": 128}]},
        { "blocks": 6, "layers": [{ "size": 3, "filters": 256}, { "size": 3, "filters": 256}]},
        { "blocks": 3, "layers": [{ "size": 3, "filters": 512}, { "size": 3, "filters": 512}]}
      ],
      "50": [
        {"blocks": 3, "layers": [{ "size": 1, "filters": 64},  { "size": 3, "filters": 64},  { "size": 1, "filters": 256}]},
        {"blocks": 4, "layers": [{ "size": 1, "filters": 128}, { "size": 3, "filters": 128}, { "size": 1, "filters": 512}]},
        {"blocks": 6, "layers": [{ "size": 1, "filters": 256}, { "size": 3, "filters": 256}, { "size": 1, "filters": 1024}]},
        {"blocks": 3, "layers": [{ "size": 1, "filters": 512}, { "size": 3, "filters": 512}, { "size": 1, "filters": 2048}]}
      ],
      "101": [
        {"blocks": 3,  "layers": [{ "size": 1, "filters": 64},  { "size": 3, "filters": 64},  { "size": 1, "filters": 256}]},
        {"blocks": 4,  "layers": [{ "size": 1, "filters": 128}, { "size": 3, "filters": 128}, { "size": 1, "filters": 512}]},
        {"blocks": 23, "layers": [{ "size": 1, "filters": 256}, { "size": 3, "filters": 256}, { "size": 1, "filters": 1024}]},
        {"blocks": 3,  "layers": [{ "size": 1, "filters": 512}, { "size": 3, "filters": 512}, { "size": 1, "filters": 2048}]}
      ],
      "152": [
        {"blocks": 3,  "layers": [{ "size": 1, "filters": 64},  { "size": 3, "filters": 64},  { "size": 1, "filters": 256}]},
        {"blocks": 8,  "layers": [{ "size": 1, "filters": 128}, { "size": 3, "filters": 128}, { "size": 1, "filters": 512}]},
        {"blocks": 36, "layers": [{ "size": 1, "filters": 256}, { "size": 3, "filters": 256}, { "size": 1, "filters": 1024}]},
        {"blocks": 3,  "layers": [{ "size": 1, "filters": 512}, { "size": 3, "filters": 512}, { "size": 1, "filters": 2048}]}
      ],
    }.get(str(layers))
    assert config is not None
    modules = []
    prev = 64
    for c in config:
      blocks = c["blocks"]
      ly = c["layers"]
      for _ in range(blocks):
        for opt in ly:
          cur = opt["filters"]
          subsample = not (cur == prev)
          prev = cur
          modules.append(ResidualBlock(**opt, subsample=subsample))
    return nn.Sequential(*modules)

if __name__ == "__main__":
  seed = os.getenv("SEED", 420)
  device = 'cpu'
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    device = 'cuda'
  elif torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)
    device = 'mps'
  print(f'Using device: {device}')

  N, C, H, W = 32, 3, 224, 224
  x = torch.randn(N, C, H, W).to(device)
  model = Resnet(layers=34).to(device)
  @timeit('forward pass completed in')
  @torch.inference_mode()
  @torch.no_grad()
  def run_inference(x): return model(x)
  for _ in range(100):
    y = run_inference(x)
    print(x.shape, y.shape)
