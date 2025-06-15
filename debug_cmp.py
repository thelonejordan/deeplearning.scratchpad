import torch

input_ids_target = torch.load("assets/input_ids_target.pt").squeeze()
input_ids = torch.load("assets/input_ids.pt")
assert input_ids_target.shape == input_ids.shape
assert input_ids.tolist() == input_ids_target.tolist()

tok_emb_target = torch.load("assets/tok_emb_target.pt")
tok_emb = torch.load("assets/tok_emb.pt")
print(tok_emb_target.shape, tok_emb.shape)
assert tok_emb_target.shape == tok_emb.shape
assert tok_emb.tolist() == tok_emb_target.tolist()

print("---"*75)

pos_emb_target = torch.load("assets/pos_emb_target.pt")
pos_emb = torch.load("assets/pos_emb.pt")

print(pos_emb_target[0].shape, pos_emb[0].shape)
assert pos_emb_target[0].shape == pos_emb[0].shape
assert torch.allclose(pos_emb[0], pos_emb_target[0])
print(pos_emb[0].flatten()[:15].tolist())
print(pos_emb_target[0].flatten()[:15].tolist())
print(torch.std_mean(pos_emb[0].flatten()))
print(torch.std_mean(pos_emb_target[0].flatten()))

print("---"*75)

print(pos_emb_target[1].shape, pos_emb[1].shape)
assert pos_emb_target[1].shape == pos_emb[1].shape
assert torch.allclose(pos_emb[1], pos_emb_target[1])
print(pos_emb[1].flatten()[-10:].tolist())
print(pos_emb_target[1].flatten()[-10:].tolist())
print(torch.std_mean(pos_emb[1].flatten()))
print(torch.std_mean(pos_emb_target[1].flatten()))

for i in range(24):
  print("---"*75)
  print("layer idx:", i)
  print()

  query_target = torch.load(f"assets/query_{i}_target.pt")
  query = torch.load(f"assets/query_{i}.pt")
  print(f"{query.shape=}")
  print(f"{query_target.shape=}")
  assert query.shape == query_target.shape
  assert torch.allclose(query, query_target)
  print(query.flatten()[-10:].tolist())
  print(query_target.flatten()[-10:].tolist())
  print(torch.std_mean(query.flatten()))
  print(torch.std_mean(query_target.flatten()))

  key_target = torch.load(f"assets/key_{i}_target.pt")
  key = torch.load(f"assets/key_{i}.pt")
  print(f"{key.shape=}")
  print(f"{key_target.shape=}")
  assert key.shape == key_target.shape
  assert torch.allclose(key, key_target)
  print(key.flatten()[-10:].tolist())
  print(key_target.flatten()[-10:].tolist())
  print(torch.std_mean(key.flatten()))
  print(torch.std_mean(key_target.flatten()))

  attn_out_target = torch.load(f"assets/attn_out_{i}_target.pt")
  attn_out = torch.load(f"assets/attn_out_{i}.pt")
  print(f"{attn_out.shape=}")
  print(f"{attn_out_target.shape=}")
  assert attn_out.shape == attn_out_target.shape
  assert torch.allclose(attn_out, attn_out_target)
  print(attn_out.flatten()[-10:].tolist())
  print(attn_out_target.flatten()[-10:].tolist())
  print(torch.std_mean(attn_out.flatten()))
  print(torch.std_mean(attn_out_target.flatten()))

  mlp_out_target = torch.load(f"assets/mlp_out_{i}_target.pt")
  mlp_out = torch.load(f"assets/mlp_out_{i}.pt")
  print(f"{mlp_out.shape=}")
  print(f"{mlp_out_target.shape=}")
  assert mlp_out.shape == mlp_out_target.shape
  assert torch.allclose(mlp_out, mlp_out_target)
  print(mlp_out.flatten()[-10:].tolist())
  print(mlp_out_target.flatten()[-10:].tolist())
  print(torch.std_mean(mlp_out.flatten()))
  print(torch.std_mean(mlp_out_target.flatten()))

  print()

  out_layer_target = torch.load(f"assets/out_layer_{i}_target.pt")
  out_layer = torch.load(f"assets/out_layer_{i}.pt")

  print(out_layer_target.shape, out_layer.shape)
  assert out_layer_target.shape == out_layer.shape
  assert torch.allclose(out_layer, out_layer_target)
  print(torch.std_mean(out_layer))
  print(torch.std_mean(out_layer_target))
  # assert torch.allclose(out_layer_1, out_layer_1_target), i
  # assert out_layer_1.tolist() == out_layer_1_target.tolist(), i

  assert mlp_out.shape == out_layer.shape
  assert torch.allclose(mlp_out, out_layer)
  assert mlp_out_target.shape == out_layer_target.shape
  assert torch.allclose(mlp_out_target, out_layer_target)

print("---"*75)

post_norm_target = torch.load(f"assets/post_norm_target.pt")
post_norm = torch.load(f"assets/post_norm.pt")[:, [-1], :]
print(f"{post_norm.shape=}")
print(f"{post_norm_target.shape=}")
assert post_norm.shape == post_norm_target.shape
assert torch.allclose(post_norm, post_norm_target)
print(post_norm.flatten()[-10:].tolist())
print(post_norm_target.flatten()[-10:].tolist())
print(torch.std_mean(post_norm.flatten()))
print(torch.std_mean(post_norm_target.flatten()))
print(torch.std_mean(post_norm - post_norm_target))

print("---"*75)

lm_head_target = torch.load(f"assets/lm_head_target.pt")#.float()
lm_head = torch.load(f"assets/lm_head.pt")[:, [-1], :]
print(f"{lm_head.shape=}")
print(f"{lm_head_target.shape=}")
print(lm_head.flatten()[-30:].tolist())
print(lm_head_target.flatten()[-30:].tolist())
print(torch.std_mean(lm_head))
print(torch.std_mean(lm_head_target))
print(torch.std_mean(lm_head - lm_head_target))
print(torch.max(torch.abs(lm_head - lm_head_target)))
assert lm_head.shape == lm_head_target.shape
assert torch.allclose(lm_head, lm_head_target)

# logits_target = torch.load("assets/logits_target.pt")
# logits_target = logits_target[:, -1, :]
# logits_target = logits_target.float()
# print(logits_target.shape)
# logits = torch.load("assets/logits.pt")
# logits = logits[:,-1,:]
# print(logits.shape)

# print(torch.argmax(logits_target))
# print(torch.argmax(logits))
# print(torch.allclose(logits, logits_target, atol=1e-3))
