# -*- coding: utf-8 -*-
"""StripedHyena custom code port for the Hugging Face Hub"""
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
from torch.nn import functional as F
from configuration_hyena import StripedHyenaConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast
from transformers.utils import logging
from typing import Optional, Tuple, Union
from model import StripedHyena
from utils import dotdict
from cache import InferenceParams
from engine import HyenaInferenceEngine
from layers import RMSNorm
from utils import dotdict, column_split, load_config_hf

logger = logging.get_logger(__name__)


class StripedHyenaPreTrainedModel(PreTrainedModel):
    config_class = StripedHyenaConfig
    base_model_prefix = "sh"
    supports_gradient_checkpointing = False
    _no_split_modules = ["AttentionBlock", "ParallelGatedConvBlock"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_missing = [r"freq"]
    _keys_to_ignore_on_load_unexpected = [r"fftconv", r"twiddle_factors"]
    _supports_flash_attn_2 = True


class StripedHyenaModelForCausalLM(StripedHyenaPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        model_config = dotdict(config.to_dict())
        self.backbone = StripedHyena(model_config)
        self.backbone.gradient_checkpointing = False
        self.config = config
        vocab_size = config.vocab_size
        if vocab_size % config.make_vocab_size_divisible_by != 0:
            vocab_size += config.make_vocab_size_divisible_by - (
                vocab_size % config.make_vocab_size_divisible_by
            )
        self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/StripedHyena-Hessian-7B",config=config,trust_remote_code=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = vocab_size
        self.post_init()
        self.force_dtype()

    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues() 
        
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.backbone.gradient_checkpointing = enable

    def get_input_embeddings(self):
        return self.backbone.embedding_layer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache:
            if self.backbone.gradient_checkpointing and self.backbone.training:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
            elif labels is not None:
                logger.warning_once(
                    "`use_cache=True` is incompatible with loss calculation. Setting `use_cache=False`..."
                )
                use_cache = False

        inputs = input_ids
        if use_cache:
            if past_key_values is None:
                past_key_values = self.backbone.initialize_inference_params()

                batch_size = input_ids.shape[0]
                past_key_values["mha"].max_batch_size = batch_size
                past_key_values["hyena"].max_batch_size = batch_size
            else:
                seqlen_offset = past_key_values["mha"].seqlen_offset
                if seqlen_offset == 0:
                    # second loop through generate will have prompt_len + 1 as seqlen
                    seqlen_offset = input_ids.shape[-1] - 1
                    past_key_values["hyena"].seqlen_offset = seqlen_offset
                    past_key_values["mha"].seqlen_offset = seqlen_offset
                else:
                    past_key_values["mha"].seqlen_offset += 1
                    past_key_values["hyena"].seqlen_offset += 1

                inputs = input_ids[
                    :,
                    -1:,
                ]

        logits, past_key_values, iir_state = self.backbone(
            inputs,
            padding_mask=attention_mask,
            inference_params_dict=past_key_values if use_cache else None,
        )

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels)

        if return_dict:
            return CausalLMOutputWithPast(
                logits=logits,
                hidden_states=None,
                past_key_values=past_key_values if use_cache else None,
                loss=loss,
            ), iir_state
        else:
            return logits, iir_state

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def encode(self, sents, **kwargs):
        embedding_list = []
        if not isinstance(sents, list):
            sents = [sents]
        for sent in sents: 
            input_ids = self.tokenizer(sent, return_tensors="pt")["input_ids"].to("cuda")
            _, state = self(input_ids)
            embedding_list.append(state)
        return embedding_list

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, past_key_values=None, **kwargs
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }
