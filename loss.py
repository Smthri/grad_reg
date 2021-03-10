import torch
from torch import nn

class GradRegLoss():
    def __init__(self, hyperparam=0):
        self.base_loss = nn.CrossEntropyLoss()
        self.hp = hyperparam

    def __call__(self, output, target, inputs, train=True):
        if train:
            cross_entropy = self.base_loss(output, target)
            grads = torch.autograd.grad(cross_entropy, inputs, create_graph=True)[0]
            reg = self.hp * torch.sum(grads ** 2) + cross_entropy
        else:
            reg = self.base_loss(output, target)

        return reg
