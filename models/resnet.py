# python3 models/resnet.py

# https://pytorch.org/vision/stable/models.html
# https://arxiv.org/abs/1512.03385
# https://d2l.ai/chapter_convolutional-modern/resnet.html
# https://github.com/tinygrad/tinygrad/blob/master/extra/models/resnet.py
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnetv2.py
# https://x.com/awnihannun/status/1832511021602500796

from typing import Optional, List
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
  expansion = 1
  def __init__(self, in_planes, planes, stride=1, groups=1, base_width=64):
    super().__init__()
    assert groups == 1 and base_width == 64, "BasicBlock only supports groups=1 and base_width=64"
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = None
    if stride != 1 or in_planes != self.expansion*planes:
      self.downsample = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion*planes)
      )

  def forward(self, x: Tensor):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    if self.downsample is not None:
      out = out + self.downsample(x)
    out = F.relu(out)
    return out


class Bottleneck(nn.Module):
  # NOTE: stride_in_1x1=False, this is the v1.5 variant
  expansion = 4
  def __init__(self, in_planes, planes, stride=1, stride_in_1x1=False, groups=1, base_width=64):
    super().__init__()
    width = int(planes * (base_width / 64.0)) * groups
    # NOTE: the original implementation places stride at the first convolution (self.conv1), control with stride_in_1x1
    self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, stride=stride if stride_in_1x1 else 1, bias=False)
    self.bn1 = nn.BatchNorm2d(width)
    self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1, stride=1 if stride_in_1x1 else stride, groups=groups, bias=False)
    self.bn2 = nn.BatchNorm2d(width)
    self.conv3 = nn.Conv2d(width, self.expansion*planes, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion*planes)
    self.downsample = None
    if stride != 1 or in_planes != self.expansion*planes:
      self.downsample = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(self.expansion*planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.downsample is not None:
      out = out + self.downsample(x)
    out = F.relu(out)
    return out

class ResNet_(nn.Module):
  def __init__(self, num, num_classes=1000, groups=1, width_per_group=64, stride_in_1x1=False):
    super().__init__()
    self.num = num
    self.block = {
      18: BasicBlock,
      34: BasicBlock,
      50: Bottleneck,
      101: Bottleneck,
      152: Bottleneck
    }[num]

    self.num_blocks = {
      18: [2,2,2,2],
      34: [3,4,6,3],
      50: [3,4,6,3],
      101: [3,4,23,3],
      152: [3,8,36,3]
    }[num]

    self.in_planes = 64

    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False, padding=3)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1, stride_in_1x1=stride_in_1x1)
    self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2, stride_in_1x1=stride_in_1x1)
    self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2, stride_in_1x1=stride_in_1x1)
    self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2, stride_in_1x1=stride_in_1x1)
    self.fc = nn.Linear(512 * self.block.expansion, num_classes)
    self.pad = nn.ZeroPad2d([1,1,1,1])

  def _make_layer(self, block, planes, num_blocks, stride, stride_in_1x1):
    strides = [stride] + [1] * (num_blocks-1)
    layers = []
    for stride in strides:
      if block == Bottleneck:
        layers.append(block(self.in_planes, planes, stride, stride_in_1x1, self.groups, self.base_width))
      else:
        layers.append(block(self.in_planes, planes, stride, self.groups, self.base_width))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x:Tensor) -> Tensor:
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.max_pool2d(self.pad(out), 3, 2)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = torch.mean(out, [2,3])
    out = self.fc(out.to(torch.float32))
    return out

class ResNet:
  def __init__(self, net: ResNet_, preprocess, categories):
    self.net = net
    self.preprocess = preprocess
    self.categories = categories

  @staticmethod
  def from_pretrained(num):
    import importlib
    assert num in (18, 34, 50, 101, 152)
    tv = importlib.import_module("torchvision.models")
    weights = tv.__dict__[f"ResNet{num}_Weights"].DEFAULT
    resnet = tv.__dict__[f"resnet{num}"]
    model = resnet(weights=weights)
    net = ResNet_(num, stride_in_1x1=True)
    net.load_state_dict(model.state_dict(), strict=True, assign=True)
    preprocess, categories = weights.transforms(), weights.meta["categories"]
    return ResNet(net, preprocess, categories)


if __name__ == "__main__":
  from torchvision.io import read_image
  model = ResNet.from_pretrained(152)
  img = read_image("downloads/images/HopperGrace300px.jpg")
  model.net.eval()
  batch = model.preprocess(img).unsqueeze(0)
  prediction = model.net(batch).squeeze(0).softmax(0)
  class_id = prediction.argmax().item()
  score = prediction[class_id].item()
  category_name = model.categories[class_id]
  print(f"{category_name}: {100 * score:.1f}%")
