import argparse, time, os
from utils_data import *
from utils_algo import *
from models import *

import numpy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(
	prog='complementary-label learning demo file.',
	usage='Demo with complementary labels.',
	description='A simple demo file with MNIST dataset.',
	epilog='end',
	add_help=True)

parser.add_argument('-lr', '--learning_rate', help='optimizer\'s learning rate', default=5e-5, type=float)
parser.add_argument('-bs', '--batch_size', help='batch_size of ordinary labels.', default=32, type=int)
# parser.add_argument('-me', '--method', help='method type. ga: gradient ascent. nn: non-negative. free: Theorem 1. pc: Ishida2017. forward: Yu2018.', choices=['ga', 'nn', 'free', 'pc', 'forward'], type=str, required=True)
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=10)
parser.add_argument('-wd', '--weight_decay', help='weight decay', default=1e-4, type=float)

def train_model(model, chosen_loss_c, optimizer, scheduler,K,ccp,num_epochs,meta_method):
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    final_epoch_acc = list()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase == 'train':
                dataloaders[phase] = complementary_train_loader
            else:
                pass
    
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss, loss_vector = chosen_loss_c(f=outputs, K=K, labels=labels, ccp=ccp, meta_method=meta_method)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if meta_method == 'ga':
                            if torch.min(loss_vector).item() < 0:
                                loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(K, requires_grad=True).view(-1,1).to(device)), 1)
                                min_loss_vector, _ = torch.min(loss_vector_with_zeros, dim=1)
                                loss = torch.sum(min_loss_vector)
                                loss.backward()
                                for group in optimizer.param_groups:
                                    for p in group['params']:
                                        p.grad = -1*p.grad
                            else:
                                loss.backward()
                        else:
                            loss.backward()
                            optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'val':
                final_element = epoch_acc.cpu().numpy()
                final_element = final_element.item()
                final_epoch_acc.append(final_element)


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # final_epoch_acc = final_epoch_acc.to_list()
    return model,final_epoch_acc

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

full_train_loader, dataloaders,ordinary_train_dataset, K,dataset_sizes = prepare_covid_data(batch_size=args.batch_size)
ordinary_train_loader, complementary_train_loader, ccp = prepare_train_loaders(full_train_loader=full_train_loader, batch_size=args.batch_size, ordinary_train_dataset=ordinary_train_dataset)

# meta_method = 'free' if args.method =='ga' else args.method

model_conv = torchvision.models.resnet18(pretrained=True)
# for param in model_conv.parameters():
#     param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 3)

model_conv = model_conv.to(device)

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv,val_acc_forward = train_model(model_conv, chosen_loss_c, optimizer_conv, exp_lr_scheduler,
                       K,ccp,num_epochs=5,meta_method='forward')

print(val_acc_forward)

model_conv,val_acc_free = train_model(model_conv, chosen_loss_c, optimizer_conv, exp_lr_scheduler,
                       K,ccp,num_epochs=5,meta_method='free')

model_conv,val_acc_ga = train_model(model_conv, chosen_loss_c, optimizer_conv, exp_lr_scheduler,
                       K,ccp,num_epochs=5,meta_method='pc')

model_conv,val_acc_nn = train_model(model_conv, chosen_loss_c, optimizer_conv, exp_lr_scheduler,
                       K,ccp,num_epochs=5,meta_method='nn')

# Data
df=pd.DataFrame({'epoch': range(0,5), 'y1_values': val_acc_forward, 'y2_values': val_acc_free, 'y3_values': val_acc_ga,'y4_values': val_acc_nn })
 
# multiple line plots
plt.plot( 'epoch', 'y1_values', data=df, marker='', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,label = 'forward')
plt.plot( 'epoch', 'y2_values', data=df, marker='', color='red', linewidth=2,label = 'free')
plt.plot( 'epoch', 'y3_values', data=df, marker='', color='green', linewidth=2, linestyle='dashed', label="pc")
plt.plot( 'epoch', 'y4_values', data=df, marker='', color='blue', linewidth=2, linestyle='dashed', label="nn")
# show legend
plt.legend()

plt.savefig("output.png")

# train_accuracy = accuracy_check(loader=dataloaders['train'], model=model)
# test_accuracy = accuracy_check(loader=dataloaders['eval'], model=model)
# print('Tr Acc: {}. Te Acc: {}.'.format(train_accuracy, test_accuracy))