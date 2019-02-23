import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import numpy as np
import cv2
import math

def get_data(directory):
    data = []
    for label in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, label)):
            for image in os.listdir(os.path.join(directory, label)):
                try:
                    x = cv2.imread(os.path.join(directory, label, image)) # Images full path
                    x = cv2.resize(x, (14, 25))
                    x = np.true_divide(x, 255)
                    # print(x)
                    # x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                    data.append((torch.Tensor(x), int(label)))
                except:
                    continue
    return data

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(25, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(5))
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(3))
        self.layer3 = nn.Sequential(
            nn.Linear(32, 11),
            # nn.BatchNorm1d(256),
            # nn.ReLU()
            )
        # self.out = nn.Linear(256, 11)
        
    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.layer3(out)
        return out

if __name__ == '__main__':
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.001
    # device = torch.device('cuda:0')
    device = torch.device('cpu')

    #instance of the Conv Net
    model = CNN().to(device)
    #loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    full_dataset = get_data("./numbers")
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    for epoch in range(num_epochs):
        total_loss = []
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        for i, (images, labels) in enumerate(train_loader):
            #Move images and labels to gpu if available
            # if cuda_avail:
            #     images = Variable(images.cuda())
            #     labels = Variable(labels.cuda())

            #Clear all accumulated gradients
            optimizer.zero_grad()
            #Predict classes using images from the test set
            outputs = model(images)
            #Compute the loss based on the predictions and actual labels
            loss = criterion(outputs,labels)
            #Backpropagate the loss
            loss.backward()

            #Adjust parameters according to the computed gradients
            optimizer.step()

            total_loss.append(loss)

        print("Average loss (epoch: " + str(epoch) + ")= " + str(sum(total_loss)/len(total_loss)))
        #Accuracy
        outputs = outputs.max(1)[1]
        correct = (outputs == labels).float().sum()
        print("Accuracy: {:.3f}".format(correct/outputs.shape[0]))

    torch.save(model.state_dict(), "./alphasmash_ocr.pth")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    for i, (images, labels) in enumerate(test_loader):
        outputs = model(images).max(1)[1]
        correct = (outputs == labels).float().sum()
        print("Accuracy: {:.3f}".format(correct/outputs.shape[0]))