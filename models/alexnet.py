# PYTHONPATH=. python models/alexnet.py

# "One weird trick for parallelizing convolutional neural networks" https://arxiv.org/abs/1404.5997
# https://pytorch.org/hub/pytorch_vision_alexnet/
# https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py

# The model that I consider is a minor variation on the winning model from the ILSVRC 2012 contest [Krizhevsky et al., 2012].
# The main difference is that it consists of one “tower” instead of two. This model has 0.2% more parameters and 2.4% fewer
# connections than the two-tower model. It has the same number of layers as the two-tower model, and the (x, y) map dimensions
# in each layer are equivalent to the (x, y) map dimensions in the two-tower model. The minor difference in parameters and
# connections arises from a necessary adjustment in the number of kernels in the convolutional layers, due to the unrestricted
# layer-to-layer connectivity in the single-tower model.(1) Another difference is that instead of a softmax final layer with
# multinomial logistic regression cost, this model's final layer has 1000 independent logistic units, trained to minimize
# cross-entropy. This cost function performs equivalently to multinomial logistic regression but it is easier to parallelize,
# because it does not require a normalization across classes.(2)
# I trained all models for exactly 90 epochs, and multiplied the learning rate by 250^(−1/3) at 25%, 50%, and 75% training progress.

# (1) In detail, the single-column model has 64, 192, 384, 384, 256 filters in the five convolutional layers, respectively
# (2) This is not an important point with only 1000 classes. But with tens of thousands of classes, the cost of normalization
#     becomes noticeable.

from __future__ import annotations
from dataclasses import dataclass, asdict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from models.helpers import timeit


@dataclass
class AlexNetConfig:
  num_classes: int = 1000
  dropout: float = 0.5


def convolution(in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, maxpool: bool=True) -> list[nn.Module]:
  layers = [
    nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
    nn.ReLU(inplace=True),
  ]
  if maxpool: layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
  return layers

def fully_connected(in_features: int, out_features: int, final_layer: bool=False, dropout: float=0.) -> list[nn.Module]:
  layers = []
  if not final_layer: layers.append(nn.Dropout(dropout))
  layers.append(nn.Linear(in_features, out_features))
  if not final_layer: layers.append(nn.ReLU(inplace=True))
  return layers


class AlexNetModel(nn.Module):
  def __init__(self, num_classes: int=1000, dropout: float=0.5):
    super().__init__()
    self.features = nn.Sequential(
      *convolution(3, 64, 11, stride=4, padding=2),
      *convolution(64, 192, 5, padding=2),
      *convolution(192, 384, 3, padding=1, maxpool=False),
      *convolution(384, 256, 3, padding=1, maxpool=False),  # NOTE: the out channels here are 256 instead of 384 as in the paper
      *convolution(256, 256, 3, padding=1),
    )
    self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    self.classifier = nn.Sequential(
      *fully_connected(256 * 6 * 6, 4096, dropout=dropout),
      *fully_connected(4096, 4096, dropout=dropout),
      *fully_connected(4096, num_classes, final_layer=True),
    )
    print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))  # should be roughly 60 million

  def forward(self, x: Tensor) -> Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

  def get_num_params(self):
    n_params = sum(p.numel() for p in self.parameters())
    return n_params


class AlexNet:
  def __init__(self, model: AlexNetModel, config: AlexNetConfig):
    self.model: AlexNetModel = model
    self.config: AlexNetConfig = config

  @property
  def device(self) -> torch.device: return next(self.model.parameters()).device

  def to(self, device: torch.device):
    self.model = self.model.to(device)
    return self

  @staticmethod
  @timeit(desc="Load time")
  def from_pretrained(num_classes: int=1000, dropout: float=0.5) -> AlexNet:
    import torchvision.models as tvm
    weights = tvm.get_weight("AlexNet_Weights.DEFAULT")
    config = AlexNetConfig(num_classes, dropout)
    model = AlexNetModel(**asdict(config))
    state_dict = weights.get_state_dict()
    model.load_state_dict(state_dict, strict=True, assign=True)
    return AlexNet(model, config)


def get_utilities():
  import torchvision.models as tvm
  weights = tvm.get_weight("AlexNet_Weights.DEFAULT")
  preprocessor, categories = weights.transforms(antialias=True), weights.meta["categories"]
  return preprocessor, list(categories)


# TODO: training script

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

  preprocessor, categories = get_utilities()
  batch = preprocessor(img).unsqueeze(0).to(device)
  classifier = AlexNet.from_pretrained(num_classes=len(categories)).to(device)
  classifier.model.eval()
  logits = classifier.model(batch)[0]
  scores = F.softmax(logits, dim=0)
  prediction_id = scores.argmax().item()
  score = scores[prediction_id].item()
  prediction = categories[prediction_id]
  print(f"prediction: {prediction}  confidence: {100 * score:.2f}%")
  label = "albatross"
  assert prediction == label, f"{prediction_id=}, {prediction=}, {label=}"
