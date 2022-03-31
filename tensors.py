# -*- coding: utf-8 -*-
import torch
import math
import numpy as np
import time


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-6

# benchmarking
iter_time = []
for_pass_time = []
backprop_time = []
update_time = []

total_s_time = time.time()
for t in range(2000):
    iter_s_time = time.time()

    # Forward pass: compute predicted y
    for_pass_s_time = time.time()
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    for_pass_e_time = time.time()
    for_pass_time.append(for_pass_e_time-for_pass_s_time)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    backprop_s_time = time.time()
    grad_y_pred = 2.0 * (y_pred - y) # TODO: why this is 2
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()
    backprop_e_time = time.time()
    backprop_time.append(backprop_e_time-backprop_s_time)

    # Update weights using gradient descent
    update_s_time = time.time()
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
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