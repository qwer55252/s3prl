import torch
from s3prl.hub import distilhubert

def get_shape(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_shape(lst[0]) if lst else []
    else:
        return []

wavs = [torch.randn(16000) for _ in range(4)]
pretrained_model = distilhubert()
results = pretrained_model(wavs)

# The representation used in the paper
representation = results["paper"]
print(f'representation: {representation.shape}')

# All hidden states
hidden_states = results["hidden_states"]
print(f'hidden_states: {get_shape(hidden_states)}')