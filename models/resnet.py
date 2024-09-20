# python3 models/resnet.py

# https://pytorch.org/vision/stable/models/resnet.html
# https://arxiv.org/abs/1512.03385
# https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
# https://d2l.ai/chapter_convolutional-modern/resnet.html
# https://github.com/tinygrad/tinygrad/blob/master/extra/models/resnet.py
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnetv2.py
# https://x.com/awnihannun/status/1832511021602500796

from typing import Union, Type, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1) -> nn.Conv2d:
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv2d:
  """1x1 convolution"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
  expansion: int = 1

  def __init__(self, in_planes: int, planes: int, stride: int=1, groups: int=1, base_width: int=64, stride_in_1x1: bool=False):
    super().__init__()
    assert groups == 1 and base_width == 64, "BasicBlock only supports groups=1 and base_width=64"
    self.conv1 = conv3x3(in_planes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = None
    if stride != 1 or in_planes != self.expansion * planes:
      self.downsample = nn.Sequential(
        conv1x1(in_planes, self.expansion * planes, stride=stride),
        nn.BatchNorm2d(self.expansion * planes)
      )

  def forward(self, x: Tensor):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += x if self.downsample is None else self.downsample(x)
    out = F.relu(out)
    return out


# NOTE: stride_in_1x1=False picks the v1.5 variant
# https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
# the original implementation places stride at the first convolution (self.conv1), control with stride_in_1x1

class Bottleneck(nn.Module):
  expansion: int = 4

  def __init__(self, in_planes: int, planes: int, stride: int=1, groups: int=1, base_width: int=64, stride_in_1x1: bool=False):
    super().__init__()
    width = int(planes * (base_width / 64.0)) * groups
    # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv1x1(in_planes, width, stride=stride if stride_in_1x1 else 1)
    self.bn1 = nn.BatchNorm2d(width)
    self.conv2 = conv3x3(width, width, stride=1 if stride_in_1x1 else stride, groups=groups)
    self.bn2 = nn.BatchNorm2d(width)
    self.conv3 = conv1x1(width, self.expansion * planes)
    self.bn3 = nn.BatchNorm2d(self.expansion * planes)
    self.downsample = None
    if stride != 1 or in_planes != self.expansion*planes:
      self.downsample = nn.Sequential(
        conv1x1(in_planes, self.expansion * planes, stride=stride),
        nn.BatchNorm2d(self.expansion * planes)
      )

  def forward(self, x: Tensor):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += x if self.downsample is None else self.downsample(x)
    out = F.relu(out)
    return out


class ResNet_(nn.Module):
  def __init__(self, num: int, num_classes: int=1000, groups: int=1, width_per_group: int=64, stride_in_1x1: bool=False):
    super().__init__()
    self.block: Type[Union[BasicBlock, Bottleneck]] = {
      18: BasicBlock,
      34: BasicBlock,
      50: Bottleneck,
      101: Bottleneck,
      152: Bottleneck
    }[num]

    self.num_blocks: List[int] = {
      18: [2,2,2,2],
      34: [3,4,6,3],
      50: [3,4,6,3],
      101: [3,4,23,3],
      152: [3,8,36,3]
    }[num]

    self.in_planes = 64
    self.groups = groups
    self.base_width = width_per_group
    self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, bias=False, padding=3)
    self.bn1 = nn.BatchNorm2d(self.in_planes)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1, stride_in_1x1=stride_in_1x1)
    self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2, stride_in_1x1=stride_in_1x1)
    self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2, stride_in_1x1=stride_in_1x1)
    self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2, stride_in_1x1=stride_in_1x1)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * self.block.expansion, num_classes)

  def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, num_blocks: int, stride: int, stride_in_1x1: bool):
    strides = [stride] + [1] * (num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride, self.groups, self.base_width, stride_in_1x1))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x: Tensor) -> Tensor:
    x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
    x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
    x = self.fc(torch.flatten(self.avgpool(x), 1))
    return x


class ResNet:
  def __init__(self, net: ResNet_, preprocess, categories):
    self.net = net
    self.preprocess = preprocess
    self.categories = categories

  @staticmethod
  def from_pretrained(variant: int):
    import importlib
    assert variant in (18, 34, 50, 101, 152)
    tv = importlib.import_module("torchvision.models")
    weights = tv.__dict__[f"ResNet{variant}_Weights"].DEFAULT
    resnet = tv.__dict__[f"resnet{variant}"]
    preprocess, categories = weights.transforms(), weights.meta["categories"]
    model = resnet(weights=weights)
    net = ResNet_(variant, len(categories))
    net.load_state_dict(model.state_dict(), strict=True, assign=True)

    return ResNet(net, preprocess, categories)


if __name__ == "__main__":
  from torchvision.io import read_image
  model = ResNet.from_pretrained(50)
  img = read_image("downloads/images/HopperGrace300px.jpg")
  model.net.eval()
  batch = model.preprocess(img).unsqueeze(0)
  prediction = model.net(batch).squeeze(0).softmax(0)
  class_id = prediction.argmax().item()
  score = prediction[class_id].item()
  category_name = model.categories[class_id]
  print(f"{category_name}: {100 * score:.1f}%")
