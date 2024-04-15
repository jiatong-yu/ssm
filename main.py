import os
import re
import numpy as np
import random
import json
from tqdm import tqdm
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoConfig

import together
import voyageai
from mistralai.client import MistralClient
together.api_key = "YOUR API KEY"
together_client = together.Together()
together.api_key = "48b71a51e3d6e6b41a2ef01bee0cac96cecdf054765bc94c54823b7e21ac84e2"
mistral_client = MistralClient(api_key="HWHxYjWcz20F5dsjJuwpoEsCeZoEdXGy")
voyage_client = voyageai.Client(api_key="pa-G6mg2OviW2rmUo2GDHubujwQ4-1TMX2nJms0oQkNBCU")

from modeling_hyena import StripedHyenaEmbedModel
from modeling_mamba import MambaEmbedModel
from model import SSMModel
from utils import _cos_sim, voyage_encode, together_encode, mistral_encode


def convert_to_gif(image_folder, output_path):
    import imageio
    def extract_layer_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else 0
    images = [img for img in sorted(os.listdir(image_folder),key=extract_layer_number) if img.endswith((".png", ".jpg"))]
    images = [os.path.join(image_folder, img) for img in images]  # Get full image paths
    frame_duration = 1
    # Create a GIF using the images
    with imageio.get_writer(output_path, mode='I', fps= 2) as writer:
        for image_path in tqdm(images):
            image = imageio.imread(image_path)
            writer.append_data(image)

def visualize_trace(prompt, channel_idx_list, layer_idxs=[33]):
    import matplotlib.pyplot as plt
    model = SSMModel("mamba")
    for layer_idx in tqdm(layer_idxs):
        _,trace = model.trace_dynamics(prompt, ssm_layer_idx=layer_idx)
        trace = trace[0]
        # trace = np.load("trace.npy")[0]
        trace = trace[:,channel_idx_list,:]
        L,D,N = trace.shape
        hori = 2 
        vert = 9
        fig, axs = plt.subplots(vert, hori, figsize=(18,18))
        for d in range(D):
            i = d // hori
            j = d % hori
            channel_idx = channel_idx_list[d]
            for n in range(N):
                axs[i][j].plot(
                    range(L), trace[:,d,n], label=f"dim {n}"
                )
            axs[i][j].set_title(f"Channel {channel_idx+1} Dynamics")
            # axs[d].legend()
            axs[i][j].set_ylim(0.15,-0.15)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"mamba, seq len {L}, layer_idx {layer_idx}")
        
        plt.savefig(f'trace_vis/trace_layer{layer_idx}.png', dpi=300)  # Saves the plot as a PNG file with a resolution of 300 DPI

def semantic_difference(args, subset="bias", name="name_birthplace",layer_idx=55):
    print(f"catagory: {subset}\nsubset: {name}")
    data = json.load(open(f"relations/{subset}/{name}.json","r"))
    template = data['prompt_templates_zs'][0]
    samples = data['samples']
    avg_sim = 0
    voyage_sim = 0 
    together_sim = 0
    if not args.api:
        model = SSMModel("mamba")
    print("WITHOUT OBJECT")
    for i in range(0, len(samples),2):
        if i >= len(samples)-1:
            break
        prompt_a = template.format(samples[i]['subject'])
        prompt_b = template.format(samples[i+1]['subject'])
        if args.api:
            enc_a = together_encode(together_client, prompt_a)
            enc_b = together_encode(together_client, prompt_b)
            together_sim += _cos_sim(enc_a, enc_b)
            enc_a = voyage_encode(voyage_client, prompt_a)
            enc_b = voyage_encode(voyage_client, prompt_b)
            voyage_sim += _cos_sim(enc_a, enc_b)
        else: 
            enc_a = model.encode(prompt_a, ssm_layer_idx=layer_idx).reshape(-1)
            enc_b = model.encode(prompt_b, ssm_layer_idx=layer_idx).reshape(-1)
            avg_sim += _cos_sim(enc_a, enc_b)
    
    print(f"no object: {avg_sim / len(samples)}")
    print(f"together: {together_sim / len(samples)}")
    print(f"voyage: {voyage_sim / len(samples)}")

    print("WITH OBJECT")
    for i in range(0, len(samples),2):
        if i >= len(samples)-1:
            break
        prompt_a = template.format(samples[i]['subject'])+" "+samples[i]['object']
        prompt_b = template.format(samples[i+1]['subject'])+" "+samples[i+1]['object']
        if args.api:
            enc_a = together_encode(together_client, prompt_a)
            enc_b = together_encode(together_client, prompt_b)
            together_sim += _cos_sim(enc_a, enc_b)
            enc_a = voyage_encode(voyage_client, prompt_a)
            enc_b = voyage_encode(voyage_client, prompt_b)
            voyage_sim += _cos_sim(enc_a, enc_b)
        else: 
            enc_a = model.encode(prompt_a, ssm_layer_idx=layer_idx).reshape(-1)
            enc_b = model.encode(prompt_b, ssm_layer_idx=layer_idx).reshape(-1)
            avg_sim += _cos_sim(enc_a, enc_b)
    
    print(f"no object: {avg_sim / len(samples)}")
    print(f"together: {together_sim / len(samples)}")
    print(f"voyage: {voyage_sim / len(samples)}")




    
if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--api",action="store_true",help="use api to encode")
    args = args.parse_args()
    semantic_difference(args)
