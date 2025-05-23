# PYTHONPATH=. python3 models/resnet.py

# https://pytorch.org/vision/stable/models/resnet.html
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# https://arxiv.org/abs/1512.03385
# https://d2l.ai/chapter_convolutional-modern/resnet.html
# https://github.com/tinygrad/tinygrad/blob/master/extra/models/resnet.py
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnetv2.py
# https://x.com/awnihannun/status/1832511021602500796

from __future__ import annotations
from typing import Union, Type, Optional, Literal, Callable, get_args
from dataclasses import dataclass, asdict
from models.helpers import timeit

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int=1, groups: int=1, dilation: int=1) -> nn.Conv2d:
  # 3x3 convolution with padding
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv2d:
  # 1x1 convolution
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
  expansion: int = 1

  def __init__(self, in_planes: int, planes: int, stride: int=1, groups: int=1, base_width: int=64, dilation: int=1,
               norm_layer: Optional[Callable[..., nn.Module]]=None, stride_in_1x1: bool=False):
    super().__init__()
    norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
    if groups != 1 or base_width != 64: raise ValueError("BasicBlock only supports groups=1 and base_width=64")
    if dilation > 1: raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    self.conv1 = conv3x3(in_planes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = None
    if stride != 1 or in_planes != self.expansion * planes:
      self.downsample = nn.Sequential(
        conv1x1(in_planes, self.expansion * planes, stride),
        norm_layer(self.expansion * planes)
      )

  def forward(self, x: Tensor):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    identity = x if self.downsample is None else self.downsample(x)
    out += identity
    out = F.relu(out)
    return out


# NOTE: stride_in_1x1=False picks the v1.5 variant
# the original implementation places stride at the first convolution (self.conv1), control with stride_in_1x1

class Bottleneck(nn.Module):
  # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
  # while original implementation places the stride at the first 1x1 convolution(self.conv1)
  # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
  # This variant is also known as ResNet V1.5 and improves accuracy according to
  # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

  expansion: int = 4

  def __init__(self, in_planes: int, planes: int, stride: int=1, groups: int=1, base_width: int=64, dilation: int=1,
               norm_layer: Optional[Callable[..., nn.Module]]=None, stride_in_1x1: bool=False):
    super().__init__()
    norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
    width = int(planes * (base_width / 64.0)) * groups
    stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)
    # Both self.conv2 and self.downsample layers downsample the input when stride != 1
    self.conv1 = conv1x1(in_planes, width, stride_1x1)
    self.bn1 = norm_layer(width)
    self.conv2 = conv3x3(width, width, stride_3x3, groups, dilation)
    self.bn2 = norm_layer(width)
    self.conv3 = conv1x1(width, self.expansion * planes)
    self.bn3 = norm_layer(self.expansion * planes)
    self.downsample = None
    if stride != 1 or in_planes != self.expansion * planes:
      self.downsample = nn.Sequential(
        conv1x1(in_planes, self.expansion * planes, stride=stride),
        norm_layer(self.expansion * planes)
      )

  def forward(self, x: Tensor):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    identity = x if self.downsample is None else self.downsample(x)
    out += identity
    out = F.relu(out)
    return out


@dataclass
class ResNetConfig:
  layers: tuple[int]
  block: Union[Type[BasicBlock], Type[Bottleneck]] = BasicBlock
  num_classes: int = 1000
  groups: int = 1
  width_per_group: int = 64
  in_planes: int = 64
  dilation: int = 1
  stride_in_1x1: bool = False

  def __post_init__(self):
    assert len(self.layers) == 4

ResNetVariant = Literal["18", "34", "50", "101", "152"]

def build(variant: ResNetVariant="18", num_classes: int=1000, stride_in_1x1: bool=False):
  assert variant in get_args(ResNetVariant), f"invalid variant: {variant}"
  block, layers = {
    "18": (BasicBlock, (2,2,2,2)),
    "34": (BasicBlock, (3,4,6,3)),
    "50": (Bottleneck, (3,4,6,3)),
    "101": (Bottleneck, (3,4,23,3)),
    "152": (Bottleneck, (3,8,36,3)),
  }[variant]
  config = ResNetConfig(layers, block, num_classes=num_classes, stride_in_1x1=stride_in_1x1)
  model = ResNetModel(**asdict(config))
  return model, config


class ResNetModel(nn.Module):
  def __init__(self, block: type[Union[BasicBlock, Bottleneck]], layers: tuple[int], num_classes: int=1000,
               zero_init_residual: bool=False, groups: int=1, width_per_group: int=64, stride_in_1x1: bool=False,
               replace_stride_with_dilation: Optional[list[bool]]=None, norm_layer: Optional[Callable[..., nn.Module]]=None, **_):
    super().__init__()
    self._norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
    self.in_planes = 64
    self.dilation = 1
    if replace_stride_with_dilation is None:
      # each element in the tuple indicates if we should replace
      # the 2x2 stride with a dilated convolution instead
      replace_stride_with_dilation = [False, False, False]
    if len(replace_stride_with_dilation) != 3:
      raise ValueError(
        "replace_stride_with_dilation should be None "
        f"or a 3-element tuple, got {replace_stride_with_dilation}"
      )
    self.groups, self.base_width = groups, width_per_group
    self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = self._norm_layer(self.in_planes)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0], stride_in_1x1=stride_in_1x1)
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], stride_in_1x1=stride_in_1x1)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], stride_in_1x1=stride_in_1x1)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], stride_in_1x1=stride_in_1x1)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # Zero-initialize the last BN in each residual branch,
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck) and m.bn3.weight is not None:
          nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
          nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

  def _make_layer(self, block: Union[Type[BasicBlock], Type[Bottleneck]], planes: int, blocks: int,
                  stride: int=1, dilate: bool=False, stride_in_1x1: bool=False) -> nn.Sequential:
    dilation = self.dilation
    if dilate: dilation, stride = dilation * stride, 1
    in_planes = planes * block.expansion
    layers = []
    for idx in range(blocks):
      _in_planes, _stride, _dilation = in_planes, 1, dilation
      if idx == 0: _in_planes, _stride, dilation = self.in_planes, stride, self.dilation
      layers.append(block(
        _in_planes,
        planes,
        stride=_stride,
        groups=self.groups,
        base_width=self.base_width,
        dilation=_dilation,
        norm_layer=self._norm_layer,
        stride_in_1x1=stride_in_1x1
      ))
    self.in_planes, self.dilation = in_planes, dilation
    return nn.Sequential(*layers)

  def forward(self, x: Tensor) -> Tensor:
    x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
    x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
    x = self.fc(torch.flatten(self.avgpool(x), 1))
    return x


class ResNet:
  def __init__(self, model: ResNetModel, config: ResNetConfig):
    self.model = model
    self.config = config

  @property
  def device(self) -> torch.device: return next(self.model.parameters()).device

  def to(self, device: torch.device):
    self.model = self.model.to(device)
    return self

  @classmethod
  @timeit(desc="Load time")
  def from_pretrained(cls, variant: ResNetVariant="18", num_classes: int=1000) -> ResNet:
    import torchvision.models as tvm
    assert variant in get_args(ResNetVariant), f"variant must be one of {get_args(ResNetVariant)}, got {variant}"
    weights = tvm.get_weight(f"ResNet{variant}_Weights.DEFAULT")
    state_dict = weights.get_state_dict()
    model, config = build(variant, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return ResNet(model, config)


def get_utilities(variant: ResNetVariant="18"):
  import torchvision.models as tvm
  weights = tvm.get_weight(f"ResNet{variant}_Weights.DEFAULT")
  preprocessor, categories = weights.transforms(antialias=True), weights.meta["categories"]
  return preprocessor, list(categories)


if __name__ == "__main__":

  import requests
  import tempfile
  import torchvision

  from helpers import set_device, set_seed
  device = set_device()
  set_seed(device)

  url = "https://upload.wikimedia.org/wikipedia/commons/1/10/070226_wandering_albatross_off_Kaikoura_3.jpg"
  with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    assert response.status_code == 200, response.content
    tmp_file.write(response.content)
    tmp_file.flush()
    img = torchvision.io.read_image(tmp_file.name)

  variant = "50"
  preprocessor, categories = get_utilities(variant)
  batch = preprocessor(img).unsqueeze(0).to(device)
  classifier = ResNet.from_pretrained(variant, num_classes=len(categories)).to(device)
  logits = classifier.model(batch)[0]
  scores = F.softmax(logits, dim=0)
  prediction_id = scores.argmax().item()
  score = scores[prediction_id].item()
  prediction = categories[prediction_id]
  print(f"prediction: {prediction}  confidence: {100 * score:.2f}%")
  label = "albatross"
  assert prediction == label, f"{prediction_id=}, {prediction=}, {label=}"
