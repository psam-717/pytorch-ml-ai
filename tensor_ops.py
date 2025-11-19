import torch

tensor = torch.rand(4,4)
ones_tensor = torch.ones(4,4)
# set the values of column with index 2 to 0
tensor[:,2] = 0
# print(tensor)

# tensor concatenation
t1 = torch.concat([tensor,ones_tensor],dim=1)
t2 = torch.concat([tensor,ones_tensor],)
# print(t1)
# print(t2)


# tensor multiplication
t3 = torch.tensor([[1,2],[3,4]])
t4 = torch.tensor([[3,5],[2,5]])
# print(t3.mul(t4))
# print(t3 * t4)

# matrix multiplication of tensors
matrix_mult = t3.matmul(t4)
matrix_mult_2 = t3 @ t4
print(matrix_mult)
print(matrix_mult_2)


# in place operations on tensors: These are operations that have a '_' suffix
# print(t4)
# print(t3)
# print(t4.copy_(t3))
# print(t4.add_(7))
