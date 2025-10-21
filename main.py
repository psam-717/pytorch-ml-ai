import torch
# import torchvision

data = [[1,2], [3,4]]
x_data = torch.tensor(data)
x_ones = torch.ones_like(x_data)
x_random = torch.rand(4,3)
print(f"Tensor data: \n {x_data} \n")
print(f"Ones tensor \n {x_ones} \n")
print(f"Random tensor \n {x_random} \n")
