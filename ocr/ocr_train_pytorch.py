import torch 
import torch.nn as nn
import os
import numpy as np
import cv2
import math

def get_data(directory):
    data, labels = [], []
    for label in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, label)):
            for image in os.listdir(os.path.join(directory, label)):
                try:
                    x = cv2.imread(os.path.join(directory, label, image)) # Images full path
                    x = cv2.resize(x, (14, 25))
                    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                    data.append(x)
                    labels.append(int(label))
                except:
                    continue
    data = np.array(data).astype(np.float16)
    data[data<255//2] = 0
    data[data>=255/2] = 1
    return (data, np.array(labels))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(5))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3))
        self.layer3 = nn.Sequential(
            nn.Linear(160, 512),
            nn.Linear(512, 11),
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
    num_epochs = 500
    batch_size = 32
    learning_rate = 0.001
    device = torch.device('cuda:0')

    #instance of the Conv Net
    cnn = CNN().to(device)
    #loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    train_dataset, train_labels = get_data("./numbers")
    train_dataset, train_labels = torch.Tensor(train_dataset).to(device), torch.Tensor(train_labels).long().to(device)
    print(train_dataset.size())
    train_dataset = train_dataset.view(train_dataset.size(0), 1, train_dataset.size(1), train_dataset.size(2))

    dat_dim = train_dataset.size()
    batches = math.ceil(dat_dim[0]/float(batch_size))
    for epoch in range(num_epochs):
        total_loss = []
        cm = np.zeros((11, 11))
        for batch in range(batches):
            if batch == batches - 1:
                images = train_dataset[batch * batch_size: ]
                labels = train_labels[batch * batch_size: ]
            else: 
                images = train_dataset[batch * batch_size : batch * batch_size + batch_size]
                labels = train_labels[batch * batch_size : batch * batch_size + batch_size]

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            # print(images.size())
            outputs = cnn(images)
            maxes = outputs.max(1)[1]
            for idx in range(len(maxes)):
                cm[maxes[idx]][labels[idx]] += 1
            loss = criterion(outputs, labels)
            total_loss.append(loss)
            loss.backward()
            optimizer.step()
        print("Average loss (epoch: " + str(epoch) + ")= " + str(sum(total_loss)/len(total_loss)))
        print('cm')
        print(cm)

    torch.save(cnn.state_dict(), "./alphasmash_ocr.pth")
