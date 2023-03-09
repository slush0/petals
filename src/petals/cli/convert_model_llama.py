"""
    FIXME WIP, not working properly yet.

    1. Use Transformers from https://github.com/zphang/transformers/tree/llama_push
    2. Convert original LLaMA weights to Hugging Face format with
           https://github.com/zphang/transformers/blob/llama_push/src/transformers/models/llama/convert_llama_weights_to_hf.py
    3. Run: PETALS_IGNORE_DEPENDENCY_VERSION=1 python -m petals.cli.convert_model_llama --model llama-7b
"""

import argparse
import os

import psutil
import torch.backends.quantized
import torch.nn as nn
import transformers
from hivemind.utils.logging import get_logger
from huggingface_hub import Repository
from tqdm.auto import tqdm
from transformers.models.bloom.modeling_bloom import BloomModel
from transformers.models.llama.modeling_llama import LLaMAModel
from transformers.models.llama.tokenization_llama import LLaMATokenizer
from petals.bloom.from_pretrained import BLOCK_BRANCH_PREFIX, CLIENT_BRANCH
from petals.client.remote_model_llama import DistributedLLaMAConfig

logger = get_logger(__file__)

DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")


def main():
    parser = argparse.ArgumentParser(description="Load bloom layers and convert to 8-bit using torch quantization.")

    parser.add_argument("--model", type=str, default="bigscience/bloom-6b3", help="Model name for from_pretrained")
    parser.add_argument("--torch_dtype", type=str, default="auto", help="Load initial model in this dtype")
    parser.add_argument("--resize_token_embeddings", type=int, default=None, help="change the vocabulary size")
    args = parser.parse_args()

    output_path = f"{args.model}-petals"

    free_ram_gb = psutil.virtual_memory().available / 2**30
    if args.model == "llama-65b" and free_ram_gb < 150:
        # FIXME this is a guess
        logger.warning(f"ACHTUNG! converting llama-65b will use up 150GB RAM, you have {free_ram_gb:.3f} free")

    assert args.torch_dtype in DTYPE_MAP, f"torch_dtype must be one of {list(DTYPE_MAP.keys())}"
    if os.path.exists(output_path) and (
        len(os.listdir(output_path)) != 0 or not os.path.isdir(output_path)
    ):
        raise FileExistsError(f"Output path {output_path} already exists and is not an empty directory")

    logger.info(f"Loading source model {args.model} (this may take a few minutes)")
    config = DistributedLLaMAConfig.from_pretrained(
        args.model,
    )
    config.dht_prefix = args.model

    model = LLaMAModel.from_pretrained(
        args.model, torch_dtype=DTYPE_MAP[args.torch_dtype]
    )
    if args.resize_token_embeddings:
        logger.info(f"Resizing token embeddings, new size = {args.resize_token_embeddings}")
        model.resize_token_embeddings(args.resize_token_embeddings)
        config.vocab_size = args.resize_token_embeddings

    tokenizer = LLaMATokenizer.from_pretrained(
        args.model,
    )

    transformer_blocks = model.layers
    logger.info(
        f"Saving transformer blocks to {output_path}/0 - {output_path}/{len(transformer_blocks)}"
    )

    os.makedirs(output_path, exist_ok=True)
    for i, block in enumerate(tqdm(transformer_blocks)):
        path = os.path.join(output_path, f"pytorch_model_block_{i}.bin")
        torch.save(block.state_dict(), path)

    logger.info(f"Saving client-side modules to {output_path}/client")

    path = os.path.join(output_path, "client")
    os.makedirs(path, exist_ok=True)

    model.layers = nn.ModuleList()
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    config.save_pretrained(path)
    logger.info(f"Converted {args.model} and stored to {output_path}")


if __name__ == "__main__":
    main()
