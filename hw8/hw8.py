import torch

x = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(0.0, requires_grad=True)
z = torch.tensor(0.0, requires_grad=True)

loss = x**2 + y**2 + z**2 - 2*x - 4*y - 6*z + 8

loss.backward()

print(f'gx={x.grad:.4f}')
print(f'gy={y.grad:.4f}')
print(f'gz={z.grad:.4f}')
