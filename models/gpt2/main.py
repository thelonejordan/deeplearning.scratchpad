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

  print("Testing generation...")
  num_return_sequences = 8
  max_length = 32
  context = "Hello, I'm a language model,"
  out = model.generate(context, max_length, num_return_sequences, top_k=50)
  print('-'*50)
  for i, sentence in enumerate(out):
    print(sentence.split('<|endoftext|>')[0])
    print('-'*50)

  print("Testing completion...")
  max_length = 200
  context = [
    "Hello, I'm a language model,",
    "Quantum computing is",
    "SpaceX and NASA have collaborated to make commercial"
  ]
  out = model.completion(context, max_length, top_k=50)
  print('-'*50)
  for i, sentence in enumerate(out):
    print(sentence.split('<|endoftext|>')[0])
    print('-'*50)


if __name__ == '__main__':
  main()
