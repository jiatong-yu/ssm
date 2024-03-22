import math
from functools import partial
import json
import os
import argparse
import numpy as np
from collections import namedtuple
from typing import List, Dict, Any, Tuple, Union
import torch
import torch.nn as nn
from mamba_ssm.models.config_mamba import MambaConfig
from transformers import AutoTokenizer
from model import MambaEmbedModel
import mteb
from mteb import MTEB
from sentence_transformers import SentenceTransformer
from model import MambaEmbedModel
import argparse
import yaml
import importlib

def _get_task_lst(config_path):
    with open(config_path, 'r') as file:
        mteb_config = yaml.safe_load(file)
    task_lst = []
    for task_category in mteb_config.values():
            if task_category['select']:
                task_lst.extend(task_category['task_names'])
    return task_lst

def main(args):
    task_lst = _get_task_lst(args.config_path)
    evaluator = MTEB(tasks=task_lst,
                     task_langs=["en"],
                    output_folder="output/vanilla")
    model = MambaEmbedModel.from_pretrained("state-spaces/mamba-2.8b", device="cuda", dtype=torch.float16)
    model.eval()
    evaluator.run(model,
                  overwrite_results=True)

def download_dataset(args):
    task_lst = _get_task_lst(args.config_path)

    for task_name in task_lst:
        print(f"Downloading {task_name} dataset")
        cls = getattr(mteb.tasks, task_name, None)
        if cls is None:
            raise ValueError(f"Task {task_name} not found in mteb.tasks")
        dataset = cls()
        dataset.load_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", default=False, action="store_true")
    parser.add_argument("--config_path", type=str, default="configs/mteb_config.yml")
    args = parser.parse_args()

    if args.download:
        download_dataset(args)
    else: 
        main(args)