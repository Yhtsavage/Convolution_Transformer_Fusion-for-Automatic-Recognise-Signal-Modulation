# import torch
# from torch import nn
# #input_size = (4, 128, 2)
# hidden_size = 6
# lstm = torch.nn.LSTM(2, hidden_size, batch_first=True, num_layers=2)
# input = torch.randn((4, 128, 2))
# out, (h, c) = lstm(input) #h.146
# print(out.shape)
# print(h.shape)
# print(c.shape)
import matplotlib.pyplot as plt
import numpy as np
#plt.rcParams['font.sans-serif'] = ['SimHei']

def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x))
    return y

def relu(x):
    return np.maximum(0,x)

def softmax(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a
    return y

plot_x = np.linspace(-100, 100, 100) # 绘制范围[-10,10]
plot_y = softmax(plot_x)
plt.plot(plot_x, plot_y)
plt.title('Softmax')
plt.show()