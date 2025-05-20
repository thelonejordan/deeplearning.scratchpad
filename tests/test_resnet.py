# PYTHONPATH=. python -m unittest tests/test_resnet.py

from typing import get_args
import requests
import unittest
import tempfile

import torch
import torchvision.models as tvm
from torchvision.io import read_image

from models.helpers import set_device
from models.resnet import ResNetVariant, ResNet

DEVICE = set_device()


def get_utilities(variant: ResNetVariant="18"):
  weights = getattr(tvm, f"ResNet{variant}_Weights").DEFAULT
  preprocessor, categories = weights.transforms(antialias=True), weights.meta["categories"]
  return preprocessor, list(categories)


class TestResNet(unittest.TestCase):

  def setUp(self):
    self._mapping = {
      v: (getattr(tvm, f"resnet{v}"), getattr(tvm, f"ResNet{v}_Weights").DEFAULT) for v in get_args(ResNetVariant)
    }
    url = "https://upload.wikimedia.org/wikipedia/commons/1/10/070226_wandering_albatross_off_Kaikoura_3.jpg"
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
      response = requests.get(url)
      tmp_file.write(response.content)
      self.img = read_image(tmp_file.name)
    # self.label = "albatross"

  def _check(self, x: torch.Tensor, y: torch.Tensor, atol=1e-8, rtol=1e-5):
    self.assertEqual(x.shape, y.shape)
    self.assertTrue(torch.allclose(x, y, atol=atol, rtol=rtol))

  def _test_variant(self, variant: str):
    preprocessor, categories = get_utilities(variant)
    batch = preprocessor(self.img).unsqueeze(0).to(DEVICE)
    builder, weights = self._mapping[variant]
    model_ref = builder(weights=weights).to(DEVICE)
    logits_ref = model_ref(batch)[0]
    del model_ref
    model = ResNet.from_pretrained(variant=variant).model.to(DEVICE)
    logits = model(batch)[0]
    del model
    self._check(logits, logits_ref)
    idx_ref = torch.nn.functional.softmax(logits_ref, dim=0).argmax().item()
    idx = torch.nn.functional.softmax(logits, dim=0).argmax().item()
    # TODO: add label check
    # self.assertEqual(categories[idx_ref], self.label)
    # self.assertEqual(categories[idx], self.label)
    self.assertEqual(idx, idx_ref)

  def test_resnet_18(self):
    self._test_variant("18")

  def test_resnet_34(self):
    self._test_variant("34")

  def test_resnet_50(self):
    self._test_variant("50")

  def test_resnet_101(self):
    self._test_variant("101")

  def test_resnet_152(self):
    self._test_variant("152")


if __name__ == "__main__":
  unittest.main()
