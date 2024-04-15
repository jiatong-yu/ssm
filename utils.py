import json
import random
import json
import os
import argparse
import torch
import datasets
from tqdm import tqdm
import time 

from math import comb
from transformers import AutoTokenizer
def _cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    source: https://github.com/embeddings-benchmark/mteb
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))[0][0]


def together_encode(client, sent, model_api = "togethercomputer/m2-bert-80M-8k-retrieval"):
    out = client.embeddings.create(input=[sent],model=model_api)
    return torch.tensor(out.data[0].embedding)

def mistral_encode(client, sent):
    out = client.embeddings(input = [sent], model="mistral-embed")
    return torch.tensor(out.data[0].embedding)

def voyage_encode(client, sent):
    out = client.embed([sent],model="voyage-2")
    return torch.tensor(out.embeddings[0])