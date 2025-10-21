import numpy as np
import torch

# tensor to numpy array
t = torch.ones(4)
n = t.numpy()

# a change in the tensor is reflected in the num
# t.add_(1)
# print(f"t: {t}")
# print(f"n: {n}")


# numpy to tensor
n = np.ones(5)
t = torch.from_numpy(n)
print(f"n: {n}")
print(f"t: {t}")