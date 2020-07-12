import torch
from clutils.models import MLP
from clutils.globals import OUTPUT_TYPE

mlp = MLP(3, [10, 20], 'cpu', 7)
mlp.output_type = OUTPUT_TYPE.OUTH
out = mlp(torch.randn(2, 3))
print(out)