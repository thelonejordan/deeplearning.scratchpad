# PYTHONPATH=. python3 models/gpt2/main.py

# from paper: Language Models are Unsupervised Multitask Learners
# https://github.com/openai/gpt-2
# https://paperswithcode.com/method/gpt-2
# checkout nanoGPT by karpathy
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
# https://github.com/tinygrad/tinygrad/blob/master/examples/gpt2.py


from models.helpers import set_device, set_seed
from models.gpt2.generate import GPT2

def main():

  device = set_device()
  set_seed(device)

  model = GPT2.from_pretrained().to(device)

  print("Testing text completion (1)...")
  num_return_sequences = 8
  context = "Hello, I'm a language model,"
  prompts = [context] * num_return_sequences
  out = model.text_completion(prompts, 64, top_k=50)
  print('-' * 50)
  for sentence in out:
    print(sentence)
    print('-' * 50)

  print()

  print("Testing text completion (2)...")
  prompts = [
    "Hello, I'm a language model,",
    "Quantum computing is",
    "SpaceX and NASA have collaborated to make commercial"
  ]
  out = model.text_completion(prompts, 256, top_k=50, top_p=0.75)
  print('-' * 50)
  for sentence in out:
    print(sentence)
    print('-' * 50)


if __name__ == '__main__':
  main()
