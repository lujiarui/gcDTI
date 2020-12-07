"""To call the train.py, use the following prompt as
    $ python train.py [0-1]
"""

import sys, os
from random import shuffle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model import GCN_CNN
from utils import *

# >>> Hyperparameter settings >>>
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

def train(model, device, train_loader, optimizer, epoch):
    """Training subroutine at each epoch(args: epoch)"""
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx,
                                                                           len(train_loader),
                                                                           100. * batch_idx / len(train_loader),
                                                                            loss.item()))

def predicting(model, device, loader):
    """Inference step"""
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


if __name__ == "__main__":
    datasets = [['davis', 'kiba'][int(sys.argv[1])]]
    cuda_name = "cuda:0"
    model_st = GCN_CNN.__name__
    # Main iterations
    for dataset in datasets:
        print('\nRunning on [' + model_st + ' ' + dataset + ']')
        processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            print('Please run create_data.py to prepare data in pytorch format!')   # Not yet preprocessing
        else:
            train_data = TestbedDataset(root='data', dataset=dataset+'_train')
            test_data = TestbedDataset(root='data', dataset=dataset+'_test')
            
            # Make data PyTorch mini-batch processing ready
            train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

            # Training model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            print('\nUtilized device name:', device, '\n')
            model = GCN_CNN().to(device)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            # Optimal recording
            best_mse = 1000
            best_ci = 0
            best_epoch = -1
            model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
            result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
            for epoch in range(NUM_EPOCHS):
                train(model, device, train_loader, optimizer, epoch+1)
                G,P = predicting(model, device, test_loader)
                ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
                if ret[1] < best_mse:
                    torch.save(model.state_dict(), model_file_name)
                    with open(result_file_name,'w') as f:
                        f.write(','.join(map(str,ret)))
                    best_epoch = epoch+1
                    best_mse = ret[1]
                    best_ci = ret[-1]
                    print('RMSE improved at epoch ', best_epoch, '; Best_MSE; Best_CI:', best_mse,best_ci, '[', model_st, dataset, ']')
                else:
                    print(ret[1],'No improvement since epoch ', best_epoch, '; Best_mse; Best_CI:', best_mse, best_ci, '[', model_st, dataset, ']')
