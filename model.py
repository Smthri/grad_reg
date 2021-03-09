import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self, BATCH_SIZE):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)
        self.act = nn.Softmax(dim=-1)

        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-04)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.bs = BATCH_SIZE

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def save(self, folder='./checkpoints/', name='model'):
        torch.save(self.state_dict(), os.path.join(folder, f'{name}.pth'))
        
    def load(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
        else:
            print(f'WARNING: path {path} does not exist!')
    
    def fit(self, trainloader, epochs, val_loader=None, callbacks=None):
        min_acc = -np.inf
        for epoch in range(epochs):  # loop over the dataset multiple times

            epoch_loss = 0.0
            epoch_acc = 0.
            val_acc = -np.inf
            val_loss = np.inf
            pbar = tqdm(enumerate(trainloader, 0))
            for i, data in pbar:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                pred = torch.argmax(outputs, axis=1)
                running_acc = (pred == labels).float().sum() / self.bs * 100
                running_loss = loss.item()

                epoch_acc += running_acc
                epoch_loss += running_loss

                # print statistics
                pbar.set_description(f'Done {(i + 1)/len(trainloader) * 100}%, Epoch: {epoch}, Loss: {running_loss}, Accuracy: {running_acc}%')
                pbar.update(self.bs)
            pbar.close()
            
            
            if not val_loader is None:
                val_acc, val_loss = self.test(val_loader)
                
            self.scheduler.step(val_loss)
            
            print(f'Avg loss: {epoch_loss / len(trainloader)}, Avg accuracy: {epoch_acc / len(trainloader)}, Val loss: {val_loss}, Val accuracy: {val_acc}')
            
            if val_acc > min_acc:
                print(f'Improved val acc from {min_acc} to {val_acc}, saving model')
                min_acc = val_acc
                self.save()
            else:
                print(f'Val acc did not improve from {min_acc}')

        print('Finished Training')

    def test(self, testloader):
        correct = 0.
        total = 0.
        loss = 0.
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self(images)
                loss += self.loss_fn(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total * 100, loss / total