"""
Code modified from https://huggingface.co/togethercomputer/StripedHyena-Hessian-7B
"""


import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer
from modeling_hyena import StripedHyenaModelForCausalLM
model_name = "togethercomputer/StripedHyena-Hessian-7B"
device = "cuda"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
config.use_cache = True
model = StripedHyenaModelForCausalLM.from_pretrained(model_name, config=config).to("cuda")
model.eval()
# print(model)
tokenizer = AutoTokenizer.from_pretrained(model_name,config=config)
tokenizer.pad_token_id = tokenizer.eos_token_id
input_ids = tokenizer("hi how are you",return_tensors="pt")["input_ids"].to("cuda")
_,state = model(input_ids)
print(state.shape)
# print(state)