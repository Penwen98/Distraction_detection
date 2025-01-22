# Libraries
import numpy as np
from tqdm import tqdm
from time import time
from PIL import Image
import datasetMng

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, Conv2d, AvgPool2d, BatchNorm2d, Dropout2d, LeakyReLU, Linear, Module
from torch.utils.data import DataLoader

from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder

# Build Model
class DistractionCNN(Module):
    def __init__(self):
        super(DistractionCNN, self).__init__()
        self.conv = Conv2d(in_channels=3, out_channels=8, kernel_size=(3,3), stride=1, padding=1)
        self.pool = AvgPool2d(kernel_size=(3,3), stride=1)
        self.relu = LeakyReLU()
        self.bn = BatchNorm2d(num_features=8)
        self.drop = Dropout2d(p=0.3)
        self.fc = Linear(in_features=8*126*126, out_features=5)
        
    def forward(self, X):
        output = self.conv(X)
        output = self.pool(output)
        output = self.relu(output)
        output = self.bn(output)
        output = self.drop(output)
        print(output.shape)
        output = output.view(-1, 8*126*126)
        output = self.fc(output)
        return output

def buildModel():
    # Device
    global device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global model, optimiser, loss_fn
    
    # Define Model and Migrate to GPU
    model = DistractionCNN().to(device)
    # Optimiser
    optimiser = Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)

    # Loss Function
    loss_fn = CrossEntropyLoss()

def trainAndEvaluate():
    for epoch in range(25):
        start = time()
        train_acc = 0
        test_acc = 0
    
        # Train
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for xtrain, ytrain in tepoch:
                optimiser.zero_grad()
                xtrain = xtrain.to(device)
                train_prob = model(xtrain)
                train_prob = train_prob.cpu()
                train_loss = loss_fn(train_prob, ytrain)
                train_loss.backward()
                optimiser.step()
                # END TRAIN
                train_pred = torch.max(train_prob, 1).indices
                train_acc += int(torch.sum(train_pred == ytrain))
                
            train_epoch_accuracy = train_acc / len_train
    
        # Evaluate
        model.eval()
        with torch.no_grad():
            for xtest, ytest in test_loader:
                xtest = xtest.to(device)
                test_prob = model(xtest)
                test_prob = test_prob.cpu()
                test_loss = loss_fn(test_prob, ytest)
                test_pred = torch.max(test_prob, 1).indices
                test_acc += int(torch.sum(test_pred == ytest))
            
            test_epoch_accuracy = test_acc / len_test
        
        end = time()
        
        diff = end - start
        
        print(f"Epoch: {epoch+1}, Time: {diff}\nTr_loss: {train_loss}, Test_loss: {test_loss}\n,Tr_acc:{train_epoch_accuracy}, Test_acc: {test_epoch_accuracy}")

