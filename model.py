import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import random
import json
from tqdm import tqdm

from transformers import AutoTokenizer, AutoConfig

from modeling_hyena import StripedHyenaEmbedModel
from modeling_mamba import MambaEmbedModel
from utils import _cos_sim

hyena_valid_layer_idxs = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

class SSMModel:
    def __init__(self, model_name: str,):
        assert model_name in ['hyena','mamba']
        self.model_name = model_name
        if model_name == 'hyena':
            model_name = "togethercomputer/StripedHyena-Hessian-7B"
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            config.use_cache = True
            model = StripedHyenaEmbedModel.from_pretrained(model_name, config=config).to("cuda")
        elif model_name == "mamba":
            model = MambaEmbedModel.from_pretrained("state-spaces/mamba-2.8b", 
                                                device="cuda",
                                                dtype=torch.float16)
            
        model.eval()
        self.model = model 

    def encode(self, sent, ssm_layer_idx, return_logits=False):
        """
        return logits, flattened encoding if return_logits is True else just the encoding
        """
        return self.model.encode(sent,ssm_layer_idx=ssm_layer_idx, return_logits=return_logits)
    
    # def patch_embedding(self, d, layer_idx=55):
    #     """
    #     Compare the difference of embedding given slight grammar but significant semantic meaning change.
    #     """
    #     template = d['r']
    #     prompts = [template.format(d['gold_s'])]
    #     for sub in d['fake_s']:
    #         prompts.append(template.format(sub))
    #     encodings = [self.encode(prompt,ssm_layer_idx=layer_idx).reshape(-1) for prompt in prompts]
    #     gold = encodings[0]
    #     avg_sim = np.average([_cos_sim(gold, enc) for enc in encodings[1:]])

    #     return avg_sim
    
    def trace_dynamics(self, sent, ssm_layer_idx,):
        """
        Trace the dynamics of a single channel of a ssm-layer given sentence.
        return logits, trace 
        """
        return self.model.trace_dynamics(
            sent, ssm_layer_idx,
        )
