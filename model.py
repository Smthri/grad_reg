import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from loss import GradRegLoss

class Net(nn.Module):
    def __init__(self, reg_param=0):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 10)
        self.act = nn.Softmax(dim=-1)

        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        print('re_param:', reg_param)
        
        self.loss_fn = GradRegLoss(reg_param)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-04)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def save(self, folder='./checkpoints/', name='model'):
        torch.save(self.state_dict(), os.path.join(folder, f'{name}.pth'))
        
    def load(self, path='./checkpoints/', name='model'):
        if os.path.exists(path):
            self.load_state_dict(torch.load(os.path.join(path, f'{name}.pth')))
        else:
            print(f'WARNING: path {path} does not exist!')