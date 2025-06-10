# Deep Learning Scratchpad

## Models

1. Language
	- [x] [GPT2 Family](models/gpt2/)
	- [x] [Llama Family](models/llama/)
	- [x] [Llama2 Family](models/llama2)
	- [x] [Llama3 Family](models/llama3/)
	- [x] [Mistral 7B](models/mistral_rolling/)
	- [x] [Mixtral 8x7B](models/mixtral.py)
	- [x] [Qwen2 Family](models/qwen2/)
	- [x] [QwQ 32B](models/qwen2/)
	- [ ] DeepSeek ...
	- [ ] Gemini ...
	- [ ] Grok

2. Vision
	- [x] [ResNet](models/resnet.py)
	- [x] [U-Net](models/unet.py)
	- [x] [AlexNet](models/alexnet.py)
	- [ ] EfficientNet
	- [ ] ViT
	- [ ] Flux
	- [ ] Stable Diffusion

3. Speech
	- [ ] Whisper

## Tests

```shell
PYTHONPATH=. python -m unittest tests
```

## Type Checking

```shell
PYTHONPATH=. python -m mypy models
```
