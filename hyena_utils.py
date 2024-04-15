import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file
from transformers import PretrainedConfig

IIR_PREFILL_MODES = [
    "recurrence",
    "modal-fft",
    "hybrid-modal-recurrence",
    "modal-scan",
    "canonical-fft",
    "iir-fir-caching",
]

# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/generation.py
@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()

@dataclass
class RecurrentInferenceParams:
    """Inference parameters passed to blocks with recurrent mode."""

    fir_filter_length: int = 3
    state_dim: int = 16
    seqlen_offset: int = 0
    fir_state_dict: dict = field(default_factory=dict)
    state_dict: dict = field(default_factory=dict)

    def reset(self):
        self.fir_filter_length = 3
        self.state_dim = 16
        self.seqlen_offset = 0

class StripedHyenaConfig(PretrainedConfig):
    model_type = "stripedhyena"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        num_filters=4096,
        inner_mlp_size=14336,
        attn_layer_idxs=[],
        hyena_layer_idxs=[],
        num_layers=32,
        tie_embeddings=False,
        short_filter_length=3,
        num_attention_heads=32,
        proj_groups=4,
        hyena_filter_groups=1,
        split_k0=True,
        column_split_hyena=True,
        column_split=False,
        model_parallel_size=1,
        pipe_parallel_size=1,
        short_filter_bias=True,
        mha_out_proj_bias=False,
        qkv_proj_bias=False,
        final_norm=True,
        use_cache=True,
        use_flash_attention_2=True,
        use_flash_rmsnorm=True,
        use_flash_depthwise=False,
        use_flashfft=False,
        inference_mode=False,
        prefill_style="fft",
        max_seqlen=32768,
        eps=1e-5,
        state_size=2,
        rotary_emb_base=500000,
        smeared_gqa=False,
        make_vocab_size_divisible_by=8,
        log_intermediate_values=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.inner_mlp_size = inner_mlp_size
        self.attn_layer_idxs = attn_layer_idxs
        self.hyena_layer_idxs = hyena_layer_idxs
        self.num_layers = num_layers
        self.tie_embeddings = tie_embeddings
        self.short_filter_length = short_filter_length
        self.num_attention_heads = num_attention_heads
        self.proj_groups = proj_groups
        self.hyena_filter_groups = hyena_filter_groups
        self.split_k0 = split_k0
        self.column_split_hyena = column_split_hyena
        self.column_split = column_split
        self.model_parallel_size = model_parallel_size
        self.pipe_parallel_size = pipe_parallel_size
        self.short_filter_bias = short_filter_bias
        self.mha_out_proj_bias = mha_out_proj_bias
        self.qkv_proj_bias = qkv_proj_bias
        self.final_norm = final_norm
        self.use_cache = use_cache
        self.use_flash_attention_2 = use_flash_attention_2
        self.use_flash_rmsnorm = use_flash_rmsnorm
        self.use_flash_depthwise = use_flash_depthwise
        self.use_flashfft = use_flashfft
        self.inference_mode = inference_mode
        self.prefill_style = prefill_style
        self.max_seqlen = max_seqlen
        self.eps = eps
        self.state_size = state_size
        self.rotary_emb_base = rotary_emb_base
        self.smeared_gqa = smeared_gqa
        self.make_vocab_size_divisible_by = make_vocab_size_divisible_by
        self.log_intermediate_values = log_intermediate_values
        super().__init__(**kwargs)

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in self.__dict__}

    @classmethod
    def from_original_config(cls, config_path, **kwargs):
        with open(config_path, "r") as f:
            config = json.load(f)

        return cls(**config, **kwargs)

def load_config_hf(model_name):
    resolved_archive_file = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    return json.load(open(resolved_archive_file))

def get_init_from_string(init_str):
    if type(init_str) == str:
        if init_str == "torch.nn.init.zeros_":
            return torch.nn.init.zeros_
        elif init_str == "torch.nn.init.xavier_uniform_":
            return torch.nn.init.xavier_uniform_
        elif init_str == "torch.nn.init.xavier_normal_":
            return torch.nn.init.xavier_normal_
        else:
            raise ValueError(f"Unrecognized init {init_str}")
        
def column_split(x, num_heads, head_size):
    """Split a tensor with `num_heads` alongside the head dimension, instead of
    across heads. Fixed to three projections
    """

    x_reshaped = x.reshape(
        x.shape[0],
        num_heads,
        3 * head_size,
    )

    x2, x1, v = (
        x_reshaped[:, :, :head_size],
        x_reshaped[
            :,
            :,
            head_size : 2 * head_size,
        ],
        x_reshaped[:, :, 2 * head_size :],
    )
    x2, x1, v = (
        x2.reshape(x2.shape[0], -1),
        x1.reshape(x1.shape[0], -1),
        v.reshape(v.shape[0], -1),
    )
    return x2, x1, v

def print_rank_0(message, debug=False, end="\n"):
    """Print from rank 0 only."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True, end=end)
    else:
        print(message, flush=True, end=end)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)

def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

class VocabUtility:
    """Split the vocabulary into `world_size` chunks amd return the
    first and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [first, last]"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size, rank, world_size):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )
    
class HyenaInferenceEngine:
    def __init__(
        self, fir_fn=None, fftconv_fn=None, iir_prefill_style="modal-fft", layer_idx=None
    ) -> None:
        self.fir_fn = fir_fn
        self.fftconv_fn = fftconv_fn
        assert (
            iir_prefill_style in IIR_PREFILL_MODES
        ), f"iir_prefill_style must be one of {IIR_PREFILL_MODES}"
        self.iir_prefill_style = iir_prefill_style
        self.layer_idx = layer_idx
        self.low_mem_mode = False

    def parallel_fir(
        self,
        fir_fn,
        u,
        weight,
        bias,
        L,
        fir_length=3,
        inference_params=None,
        prefill_mode=None,
        padding_mask=None,
    ):
        """Compute the output state of the long convolutional filter."""
        # prepare input layout, dimensions and dispatch to fir kernel
        if fir_fn != torch.nn.functional.conv1d:
            z_pre = fir_fn(u)[:, :L]  # B, L, D
            z_pre = z_pre.permute(0, 2, 1)
        else:
            u = u.permute(0, 2, 1)  # B, D, L
            z_pre = fir_fn(
                u,
                weight,
                bias,
                stride=1,
                padding=fir_length - 1,
                groups=u.shape[1],
            )[..., :L]

        # handle padding post fir, the only place with biases
        if type(padding_mask) == torch.Tensor:
            z_pre = z_pre * padding_mask[:, None]

        if inference_params is not None:
            # handle seqlen last and dim last cases for `u`
            if fir_fn != torch.nn.functional.conv1d:
                fir_state = u[:, -fir_length + 1 :].permute(0, 2, 1)
            else:
                fir_state = u[..., -fir_length + 1 :]
        else:
            fir_state = None

        return z_pre, fir_state

    def parallel_iir(
        self,
        z_pre,
        h,
        D,
        L,
        poles,
        t,
        dims,
        layer_idx,
        inference_params=None,
        prefill_style="fft",
        fftconv_fn=None,
        padding_mask=None,
        use_flashfft=False,
        column_split_hyena=False,
        long_fir_threshold=None,
    ):
        """Compute the output state of the short convolutional filter."""
        fft_size = 2 * L
        hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
        # Compatibility with training infra that column splits the projections
        if column_split_hyena:
            z = z_pre.reshape(
                z_pre.shape[0],
                num_attention_heads,
                3 * hidden_size_per_attention_head,
                z_pre.shape[2],
            )
            x2, x1, v = (
                z[:, :, :hidden_size_per_attention_head],
                z[
                    :,
                    :,
                    hidden_size_per_attention_head : 2 * hidden_size_per_attention_head,
                ],
                z[:, :, 2 * hidden_size_per_attention_head :],
            )
            x2, x1, v = (
                x2.reshape(x2.shape[0], -1, x2.shape[-1]),
                x1.reshape(x1.shape[0], -1, x1.shape[-1]),
                v.reshape(v.shape[0], -1, v.shape[-1]),
            )
        else:
            x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)

        x1v = x1 * v

        if use_flashfft and (L % 2) == 0:  # only works with even L
            y = fftconv_fn(
                x1v.to(dtype=torch.bfloat16).contiguous(),
                h.to(dtype=torch.float32),
            )
            X_s = None

        elif long_fir_threshold is None:
            H = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
            X_s = torch.fft.fft(x1v.to(dtype=torch.float32), n=fft_size)
            X = X_s[..., : H.shape[-1]]
            if len(z_pre.shape) > 3:
                H = H.unsqueeze(1)
            y = torch.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]
        else:
            assert h.shape[0] == 1, "batch size must be 1 for long_fir_threshold"
            h = h[0][:, None]  # rearrange to d, 1, l for depthwise conv1d
            h = h[..., :long_fir_threshold]
            y = F.conv1d(
                x1v,
                h.to(dtype=x1v.dtype),
                stride=1,
                groups=x1v.shape[1],
                padding=h.shape[-1] - 1,
            )[..., :L]

        y = y.to(dtype=x1v.dtype)
        y = (y + x1v * D.unsqueeze(-1)) * x2
        if inference_params is not None:
            if prefill_style == "fft":
                self.prefill_via_modal_fft(
                    inference_params=inference_params,
                    x1v=x1v,
                    X_s=X_s,
                    L=L,
                    t=t,
                    poles=poles,
                    dims=dims,
                    layer_idx=layer_idx,
                    use_flashfft=use_flashfft,
                )

            elif prefill_style == "recurrence":
                self.prefill_via_direct_recurrence(
                    inference_params=inference_params,
                    x1v=x1v,
                    L=L,
                    poles=poles,
                )

            else:
                raise NotImplementedError
            if self.low_mem_mode:
                del z_pre, x2, x1, v, x1v, h
                torch.cuda.empty_cache()

        return y.permute(0, 2, 1)

    def step_fir(self, u, fir_state, weight, bias=None):
        """Step the FIR filter.

        Note:
        `fir_state` contains the last `short_filter_length - 1` elements of `u`: `u_(L-2), u_{L-1), ...`
        We assume dimensions of `short_filter_weight` to be `[d, 1, short_filter_len]` (SISO / multi SISO layout).
        """
        h0, h = weight[..., 0, -1], weight[..., 0, :-1]
        h0, h = h0[None], h[None]
        y = h0 * u + torch.sum(fir_state * h, dim=-1) + bias

        # update
        fir_state = torch.roll(fir_state, -1, dims=2)
        fir_state[..., -1] = u
        return y, fir_state

    def step_iir(self, x2, x1, v, D, residues, poles, iir_state, iir_groups=1):
        x1v = x1 * v

        residues, poles = (
            torch.view_as_complex(residues.to(torch.float32)),
            torch.view_as_complex(poles.to(torch.float32)),
        )
        # squeeze the dummy seqlen dimension
        # D, state_dim, 1 -> 1, D, state_dim
        residues, poles = residues[..., 0][None], poles[..., 0][None]
        iir_state = poles * iir_state + x1v[..., None]

        res_state = torch.sum(residues * iir_state, dim=-1).real

        if iir_groups > 1:
            raise NotImplementedError
        y = x2 * (res_state + D * x1v)

        return y, iir_state

    def prefill_via_fir_caching(self, u, inference_params, L, *args, **kwargs):
        """Turns the IIR filter into a FIR and uses a cache for decoding."""
        raise NotImplementedError(":)")

    def prefill_via_direct_recurrence(self, inference_params, x1v, L, poles, *args, **kwargs):
        """
        Compute the IIR state via explicit SSM recurrence (modal form)
        """
        x1v_ = x1v[..., None, None]  # b, d, l, sdim, reim
        x1v_ = x1v_.repeat(1, 1, 1, 1, 2)  # b, d, l, sdim, reim

        state = x1v_[:, :, 0]
        poles = poles[:, :, 0].to(dtype=torch.float32)

        for i in range(L):
            state = poles * state + x1v_[:, :, i]
        inference_params.state_dict[self.layer_idx] = torch.view_as_complex(
            state.to(dtype=torch.float32)
        )

    def prefill_via_hybrid_recurrence(
        self, inference_params, u, log_poles, x1v_f_a, L, *args, **kwargs
    ):
        """
        Compute the IIR state via hybrid recurrence-convolution over blocks
        """
        raise NotImplementedError(":)")

    def prefill_via_scan(self, u, inference_params=None, *args, **kwargs):
        raise NotImplementedError

    def prefill_via_canonical_fft(self, u, inference_params=None, *args, **kwargs):
        """
        Compute the IIR state via a single FFT with the denominator of the SSM in companion form.

        This is the most memory efficient "parallelized" prefilling method for Hyena.

        From: https://arxiv.org/abs/2310.18780
        """
        raise NotImplementedError(":)")
    # entry here.
    def prefill_via_modal_fft(
        self,
        inference_params,
        x1v,
        L,
        poles,
        t,
        dims,
        layer_idx,
        X_s=None,
        use_flashfft=False,
        state_dtype=torch.complex64,
        *args,
        **kwargs,
    ):
        """
        Compute the IIR state via a single FFT, using the poles of the SSM in modal form.
        """
        # When the model has a long convolution derived from a SSM in modal form and prefill_style is "fft",
        # we split the filter into poles and residues and reuse FFT computation on the input.
        # This optimization is currently not supported when using flashfftconv.
        hidden_size, _, _, state_size, hyena_filter_groups = dims

        if use_flashfft:
            # using real states
            poles = poles.squeeze().reshape(poles.shape[0], -1)[..., None]

            state_s = poles**t
            if hyena_filter_groups > 1:
                raise NotImplementedError

            x1v = x1v[:, :, None].repeat(1, 1, 2 * state_size, 1)
            x1v = x1v.reshape(x1v.shape[0], -1, x1v.shape[-1])
            state_s = state_s[None]

            state = self.fftconv_fn(
                x1v.contiguous(),
                state_s.to(dtype=torch.float32),
            )
            state = state[..., L - 1].reshape(x1v.shape[0], hidden_size, state_size, 2)
            state = torch.view_as_complex(state.contiguous())
            inference_params.state_dict[self.layer_idx] = state.to(dtype=state_dtype)
        else:
            # here. TODO try use flashfft?
            assert X_s is not None
            bs = x1v.shape[0]
            fft_size = 2 * L
            poles = torch.view_as_complex(poles.to(torch.float32))
            state_s = poles**t
            state_S = torch.fft.fft(state_s, n=fft_size).repeat(
                bs, 1, 1, 1
            )  # B, D, state_dim, 2 * L
            if hyena_filter_groups > 1:
                state_S = state_S.repeat_interleave(hidden_size // hyena_filter_groups, 1)
            state = torch.fft.ifft(X_s[..., None, :] * state_S, n=fft_size)
            inference_params.state_dict[layer_idx] = state[..., L - 1].to(dtype=state_dtype)

    def _compute_state(self, log_poles, u, t, L, *args, **kwargs):
        """
        Compute the IIR state given an input `u` and log_poles of the modal system.
        """
        bs = u.shape[0]
        fft_size = 2 * L
        U = torch.fft.rfft(u.to(torch.float32), n=fft_size)
        fft_size = 2 * L
        x = (log_poles * t).exp()
        # [batch, hidden_size, state_dim, 2 * seqlen]
        X = torch.fft.fft(x, n=fft_size).repeat(bs, 1, 1, 1)
        state = torch.fft.ifft(U[..., None, :] * X, n=fft_size)[..., :L]
        return state

class RMSNorm(torch.nn.Module):
    def __init__(self, config):
        super(RMSNorm, self).__init__()
        self.eps, self.hidden_size = config.eps, config.hidden_size
        self.scale = torch.nn.Parameter(torch.ones(self.hidden_size))
        self.register_parameter("scale", self.scale)
        self.use_flash_rmsnorm = config.get("use_flash_rmsnorm", False)

        if self.use_flash_rmsnorm:
            try:
                from flash_attn.ops.rms_norm import rms_norm as rmsnorm_func

                self.rmsnorm_func = rmsnorm_func
            except:
                raise ImportError(
                    "For `use_flash_rmsnorm`: `pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/layer_norm`"
                )

    def forward(self, x):
        if self.use_flash_rmsnorm:
            return self.rmsnorm_func(x, self.scale, self.eps)
        else:
            y = x / (x.norm(2, dim=-1, keepdim=True) * self.hidden_size ** (-1.0 / 2) + self.eps)
            return self.scale * y

class ParallelGatedMLP(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        multiple_of = config.get("inner_size_multiple_of", 64)
        self.act = F.silu

        self.multiple_of = multiple_of * config.model_parallel_size

        inner_size = int(2 * config.hidden_size * 4 / 3)
        inner_size = self.multiple_of * ((inner_size + self.multiple_of - 1) // self.multiple_of)
        # if specified in the config, inner_size will be used instead of the calculated value
        if config.get("inner_mlp_size", None) is not None:
            inner_size = config.inner_mlp_size

        self.l1 = nn.Linear(
            in_features=config.hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=config.hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=config.hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        if type(z1) == tuple:
            z1 = z1[0]
        if type(z2) == tuple:
            z2 = z2[0]
        y = self.l3(self.act(z1) * z2)
        return y[0] if type(y) == tuple else y

class Embedding(nn.Module):
    _train_dtype = "bf16"

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)

    def embed(self, input_ids, position_ids=None, tokentype_ids=None):
        embeddings = self.word_embeddings(input_ids)
        return embeddings

    def unembed(self, u):
        weight = self.word_embeddings.weight
        return torch.matmul(u, weight)

class VocabParallelEmbedding(nn.Embedding):
    "Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/embedding.py"

    def __init__(self, config):
        vocab_size, process_group, padding_idx = (
            config.vocab_size,
            config.get("process_group", None),
            config.get("padding_idx", None),
        )
        self.process_group = process_group
        if process_group is not None:
            world_size = torch.distributed.get_world_size(process_group)
            if vocab_size % world_size != 0:
                raise ValueError(
                    f"vocab_size ({vocab_size}) must be divisible by " f"world_size ({world_size})"
                )
            if world_size > 1 and padding_idx is not None:
                raise RuntimeError("ParallelEmbedding does not support padding_idx")
        else:
            world_size = 1
        super().__init__(
            vocab_size // world_size,
            embedding_dim=config.hidden_size,
            padding_idx=padding_idx,
        )

    def embed(self, x: Tensor) -> Tensor:
        if self.process_group is None:
            return self.forward(x)
        else:
            rank = torch.distributed.get_rank(self.process_group)
            vocab_size = self.num_embeddings
            vocab_start_index, vocab_end_index = (
                rank * vocab_size,
                (rank + 1) * vocab_size,
            )
            # Create a mask of valid vocab ids (1 means it needs to be masked).
            input_ids_mask = (x < vocab_start_index) | (x >= vocab_end_index)
            x = x - vocab_start_index
            x[input_ids_mask] = 0
            embeddings = self.forward(x)
            embeddings[input_ids_mask] = 0.0
            # Reduce to the global process group
            torch.distributed.all_reduce(embeddings, group=self.process_group)
            return embeddings

    def unembed(self, u: Tensor) -> Tensor:
        if self.process_group is None:
            return u @ self.weight.T
        else:
            raise NotImplementedError