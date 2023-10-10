# srf-attention
Simplex Random Feature attention, in PyTorch

## A Prelude
### Why? What? Huh?
Softmax attention ate the world. But now it's eating our wallets. Luckily enough for us wordcels, those nifty shape rotators realized that even though softmax isn't stationary, it's amenable to Monte Carlo methods. Translation: we can retrofit pretrained LLMs for recurrent inference, with a little fine tuning! Smarter men than I proceeded to publish [this](https://arxiv.org/abs/2009.14794), [this](https://arxiv.org/abs/2205.15317), and [that](https://arxiv.org/abs/2301.13856). This repo is a PyTorch implementation of "that", with some syntactic sugar added to aid digestion. Just drop the Attention module into your code, in place of your SDP implementation, and fine tune under the ordinary training objective.

### What is this good for?
Dropping that pesky KV cache from $`O(LD)`$ to $`O(D^2)`$!

### Next steps
First, do the [appropriate](#Usage) model surgery. Then, resume the original training objective. [Here's](https://huggingface.co/datasets/reversebutlerianjihad/AnorexicPajama) a dataset we used internally for a Llama 2 retrofit that's now in production.

## Installation
```bash
pip install git+https://github.com/notarussianteenager/srf-attention
```

## Usage
```python
import torch
from srf_attention import Attention

device = 'cpu'

B, H, L, D = (1, 8, 1024, 128)

q, k, v = [torch.randn(B, H, L, D) for _ in range(3)]

# CHUNK_SIZE controls the memory consumption of the attention computation
CHUNK_SIZE=256

# Simplex Random Feature (SRF) Attention module
# All intermediate computations done in FP32, but cached values are FP16.
# Recomputes the attention matrix in the backward pass instead of storing it:
attn = Attention(d=D, n_features=D, causal=True, device=device)

# Use 1 instance for each layer,
# and disable auto-redraw of random features prior to beginning training:
attn.redraw_on_call_(False)

# During fine-tuning, replace your softmax attention function with this:
o = attn(q, k, v, mode='train', attn_fn='torch', chunk_size=CHUNK_SIZE)

# On each training step, call redraw_() FIRST to resample the random features:
attn.redraw_()

# That's it! Now just fine-tune.
```

## Example
Here's an example, using the HF Transformers [diff](https://github.com/notarussianteenager/transformers-llama-srf) we wrote to retrofit Llama 2 with SRF attention:
```python
# Make sure TILE_SIZE env var is set, we use TILE_SIZE=256
import torch
# install using `pip install git+https://github.com/notarussianteenager/transformers-llama-srf`
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')


### Enable SRF,
### Disable random feature auto-redraw
for module in model.modules():
  if isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention):
    module.use_fast_attn_(True)
    module.attn_fn.redraw_on_call_(False)

### Utility function for resampling features
def resample_rfs(model):
  for module in model.modules():
    if isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention):
      module.attn_fn.redraw_()

### Pseudo-code:
optimizer = YourOptimizerHere()
for step, batch in enumerate(imaginary_dataset):
  inputs, targets = batch
  # Always resample random features manually,
  # because auto-resampling causes issues with checkpointing
  resample_rfs(model)
  outputs = model(inputs)
  logits = outputs.logits.reshape(-1, outputs.logits.shape[-1])
  loss = torch.nn.functional.cross_entropy(logits, targets['input_ids'].reshape(-1))
  loss.backward()
  optimizer.step()
```
