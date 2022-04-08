# -*- coding: utf-8 -*-
import torch
import math
import numpy as np
import time

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Create random Tensors for weights. For a third order polynomial, we need
# 4 weights: y = a + b x + c x^2 + d x^3
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

# benchmarking
iter_time = []
for_pass_time = []
backprop_time = []
update_time = []

total_s_time = time.time()
for t in range(2000):
    iter_s_time = time.time()

    # Forward pass: compute predicted y using operations on Tensors.
    for_pass_s_time = time.time()
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    for_pass_e_time = time.time()
    for_pass_time.append(for_pass_e_time-for_pass_s_time)

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    backprop_s_time = time.time()
    loss.backward()
    backprop_e_time = time.time()
    backprop_time.append(backprop_e_time-backprop_s_time)

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    update_s_time = time.time()
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Manually zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
    update_e_time = time.time()
    update_time.append(update_e_time-update_s_time)

    iter_e_time = time.time()
    iter_time.append(iter_e_time-iter_s_time)
total_e_time = time.time()
total_time = total_e_time-total_s_time

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
print()

# benchmark
print("mean iter time:", np.mean(iter_time))
print("mean forward pass time:", np.mean(for_pass_time))
print("mean backprop time:", np.mean(backprop_time))
print("mean update weight time:", np.mean(update_time))
print("total time:", total_time)