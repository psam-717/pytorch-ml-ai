import torch 

# creating tensors and tracking operations for gradient computation
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# define a vector valued function with vectors 'a' and 'b'
Q = 3*a**3 - b**2

external_grad = torch.tensor([2., 1.])
Q.backward(gradient=external_grad) # perform differentiation

# check if collected gradient are correct
print(18*a**2 == a.grad)
print(-4*b == b.grad)