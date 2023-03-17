# To reproduce the issue, download shard.bin into this directory and run "python test_int8.py".
# It will prints "int8" on the last line, while float16 is expected.
import torch
from petals.utils.convert_block import convert_block
from transformers import LlamaConfig

block = torch.load(open('shard.bin', 'rb'))
config = LlamaConfig()
print(config)
print(block)

block2 = convert_block(block, config, [torch.device('cuda:0')],
            'cuda:0', True, 6.0, True)
print(block2)
print(next(block2.parameters()).dtype)
