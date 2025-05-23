# PYTHONPATH=. python -m unittest tests/test_alexnet.py

import requests
import unittest
import tempfile

import torch
import torchvision.models as tvm
from torchvision.io import read_image

from models.helpers import set_device
from models.alexnet import AlexNet, get_utilities

DEVICE = set_device()


class TestAlexNet(unittest.TestCase):
  def setUp(self):
    url = "https://upload.wikimedia.org/wikipedia/commons/1/10/070226_wandering_albatross_off_Kaikoura_3.jpg"
    with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
      headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
      response = requests.get(url, headers=headers)
      if response.status_code == 200:
        self.skip = False
        tmp_file.write(response.content)
        tmp_file.flush()
        self.img = read_image(tmp_file.name)
      else:
        print(response.content)
        self.skip = True
    self.label = "albatross"

  def _check(self, x: torch.Tensor, y: torch.Tensor, atol=1e-8, rtol=1e-6):
    self.assertEqual(x.shape, y.shape)
    self.assertTrue(torch.allclose(x, y, atol=atol, rtol=rtol))

  def _test_variant(self):
    if self.skip: self.skipTest(f"image download failed")
    preprocessor, categories = get_utilities()
    batch = preprocessor(self.img).unsqueeze(0).to(DEVICE)
    model_ref = tvm.alexnet(weights=tvm.AlexNet_Weights.DEFAULT).to(DEVICE)
    model_ref.eval()
    logits_ref = model_ref(batch)[0]
    del model_ref
    model = AlexNet.from_pretrained(len(categories)).to(DEVICE).model
    model.eval()
    logits = model(batch)[0]
    del model
    self._check(logits, logits_ref)
    idx_ref = torch.nn.functional.softmax(logits_ref, dim=0).argmax().item()
    idx = torch.nn.functional.softmax(logits, dim=0).argmax().item()
    self.assertEqual(categories[idx_ref], self.label, f"exp: {self.label}, got: {categories[idx_ref]}")
    self.assertEqual(categories[idx], self.label, f"exp: {self.label}, got: {categories[idx_ref]}")
    self.assertEqual(idx, idx_ref, f"exp: {idx}, got: {idx_ref}")

  def test_alexnet(self):
    self._test_variant()


if __name__ == "__main__":
  unittest.main()
