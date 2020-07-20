import torch
import clutils
from clutils.models import MLP

mlp = MLP(3, [10, 20], 'cpu', 7)
mlp.output_type = clutils.OUTPUT_TYPE.LAST_OUT
out = mlp(torch.randn(2, 3))
print(out)