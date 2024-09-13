# python3 models/unet.py

# https://arxiv.org/abs/1505.04597
# https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
# https://paperswithcode.com/method/u-net
# https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical

import math
from dataclasses import dataclass

import torch
from torch import nn, Tensor

class Convolution(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.model = nn.Sequential(
      nn.Conv2d(in_c, out_c, kernel_size=3),
      # nn.BatchNorm2d(out_c),
      nn.ReLU(),
      nn.Conv2d(out_c, out_c, kernel_size=3),
      # nn.BatchNorm2d(out_c),
      nn.ReLU(),
    )

  def forward(self, x: Tensor):
    return self.model(x)

class Encoder(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv = Convolution(in_c, out_c)
    self.downsample = nn.MaxPool2d(2) # stride defaults to kernel size

  def forward(self, x: Tensor):
    c = self.conv(x)
    p = self.downsample(c)
    return c, p

class Decoder(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.upsample = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
    self.conv = Convolution(2*out_c, out_c)

  def forward(self, x: Tensor, skip: Tensor):
    x = self.upsample(x)
    hrem, wrem = (skip.size(-2)-x.size(-2))//2, (skip.size(-1)-x.size(-1))//2
    cropped = skip[..., hrem:(hrem+x.size(-2)), wrem:(wrem+x.size(-1))]
    x = torch.cat([x, cropped], dim=1)
    x = self.conv(x)
    return x

class UNet(nn.Module):
  def __init__(self, in_c: int=1, out_c: int=2):
    super().__init__()
    # encoder
    self.e1 = Encoder(in_c, 64)
    self.e2 = Encoder(64, 128)
    self.e3 = Encoder(128, 256)
    self.e4 = Encoder(256, 512)
    # bottleneck
    self.b = Convolution(512, 1024)
    # decoder
    self.d1 = Decoder(1024, 512)
    self.d2 = Decoder(512, 256)
    self.d3 = Decoder(256, 128)
    self.d4 = Decoder(128, 64)
    # classifier
    self.classifier = nn.Conv2d(64, out_c, kernel_size=1)

    self.apply(self.init_weights)

  def init_weights(self, m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
      N = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
      nn.init.normal_(m.weight, std=math.sqrt(2.0 / N))

  def forward(self, x: Tensor):
    # encoder
    s1, p1 = self.e1(x)
    s2, p2 = self.e2(p1)
    s3, p3 = self.e3(p2)
    s4, p4 = self.e4(p3)
    # bottleneck
    b = self.b(p4)
    # decoder
    d1 = self.d1(b, s4)
    d2 = self.d2(d1, s3)
    d3 = self.d3(d2, s2)
    d4 = self.d4(d3, s1)
    # classifier
    out = self.classifier(d4)
    return out

@dataclass
class TrainConfig:
  momentum: float = 0.99

def train(X_train: Tensor, Y_train: Tensor, config: TrainConfig=None):
  config = TrainConfig() if config is None else config
  # TODO: impplement this
  raise NotImplementedError()

if __name__ == "__main__":
  from helpers import set_device, set_seed
  device = set_device()
  set_seed(device)

  N, C, H, W = 2, 3, 572, 572
  sample_input = torch.randn(N, C, H, W)
  model = UNet(C).to(device)
  # cmodel = torch.compile(model).to(device)
  out = model(sample_input.to(device))
  print(sample_input.shape, out.shape)
