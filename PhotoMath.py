import torchvision
import numpy as np
import os
import pandas as pd
import matplotlib.image as img
from torchvision import transforms
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.models as models

class MathDataset(Dataset):
    def __init__(self, data, path, transform=None):
        self.dataset = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_name,label = self.dataset[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(1620, 512)
        self.fc2 = nn.Linear(512, 12)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

categories = []
fileName = []
filename1 = []
categories1 = []
inputFiles = os.listdir("../../Downloads/extracted_images")
for folder in inputFiles:
    brojac = 0
    insideFolder = os.listdir("../../Downloads/extracted_images/" + str(folder))
    vrijednost = str(folder)
    if vrijednost == "+":
        vrijednost = 10
    elif vrijednost == "-":
        vrijednost = 11
    for f in insideFolder:
        if brojac < 3500:
            fileName.append(str(folder) + "/" + str(f))
            categories.append(int(vrijednost))
            brojac += 1
        else:
            break


df = pd.DataFrame({
    'filename': fileName,
    'categories': categories
})


transform =transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1),
                               transforms.RandomRotation(degrees=(-7,7), fill=(0,)),
                              transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train,valid = train_test_split(df,test_size=0.1)
train_data = MathDataset(train,"../../Downloads/extracted_images/",transform)
valid_data = MathDataset(valid  ,"../../Downloads/extracted_images/",transform)

epochs = 15
classes = 12
batch = 65
learning_rate = 0.001

train_loader = DataLoader(dataset=train_data,batch_size=batch,shuffle=True,num_workers=0)
valid_loader = DataLoader(dataset=valid_data,batch_size=batch,shuffle=True,num_workers=0)


device = torch.device('cuda')
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

train_losses = []
valid_losses = []

PATH = "photomath.pth"

for epoch in range(1, epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0
    correct_train = 0
    correct_valid = 0

    model.train()
    for data, target in train_loader:


        # move data to gpu
        data = data.to(device)
        target = target.to(device)

        # set gradients to zero
        optimizer.zero_grad()
        # forward pass
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        correct_train += (predicted == target).sum().item()
        # calculate cross entropy loss
        loss = criterion(output, target)
        # backpropagation
        loss.backward()
        optimizer.step()
        # update training loss
        train_loss += loss.item() * data.size(0)

    model.eval()
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        _, predicted = torch.max(output.data, 1)
        correct_valid += (predicted == target).sum().item()

        loss = criterion(output, target)

        # update validation loss
        valid_loss += loss.item() * data.size(0)


    # calculate average loss for epoch
    train_loss = train_loss / len(train_loader.sampler)
    correct_train = correct_train / len(train_loader.sampler)

    valid_loss = valid_loss/len(valid_loader.sampler)
    correct_valid = correct_valid/len(valid_loader.sampler)
    train_losses.append(train_loss)

    # print-training/validation-statistics
    print(
        'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tCorrect training: {} \tCorrect valid: {}'.format(
            epoch, train_loss, valid_loss, correct_train, correct_valid))

    torch.save(model.state_dict(), PATH)

torch.save(model.state_dict(), PATH)