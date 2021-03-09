import torch
from torch import nn

class GradRegLoss():
    def __init__(self):
        self.base_loss = nn.CrossEntropyLoss()
        
    def __call__(self, output, target):
        return self.base_loss(output, target)
