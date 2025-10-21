import torch

# define the shape of the tensor
shape = (4, 3)
random_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random 4 x 3 tensor: \n {random_tensor} \n")
print(f"Ones 4 x 3 tensor: \n {ones_tensor} \n")
print(f"Zeros 4 x 3 tensor: \n {zeros_tensor} \n")
 