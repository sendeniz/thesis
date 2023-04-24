#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 20:21:34 2022

@author: sen
"""
import torch
from torch.utils.data.dataset import T_co
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from rnncells import RnnCell, GruCell, LstmCell
from rnns import SimpleRNN, GruRNN, LstmRNN
import torch.nn as nn
import torch.nn.functional as F
from utils import top1accuracy, top5accuracy, strip_square_brackets
from numpy import genfromtxt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_decay = 0.0005
sequence_length = 28 * 28
input_size = 28
hidden_size = 64
nlayers = 2
nclasses = 10
warmup = 5
batch_size = 64
nepochs = 100
T_max = nepochs - warmup
# defines number of runs or repititions of the experiment
nruns = 5

save_model = True
#continue_training = True

data_dir =  'data/'
mnist = True
cifar10 = False

init_simplernn = False
init_grurnn = False
init_lstmrnn = True
init_hippornn = False

model_names = ['simplernn_', 
                'grurnn_',
                'lstmrnn_',
                'hippornn_']

dataset_names = ['smnist_', 
                'scifar_']

if init_simplernn == True:
    lr = 0.00001
    current_model = model_names[0]
    path_cpt_file = f'cpts/{current_model}smnist.cpt'
    model_name = 'Simple RNN'
elif init_grurnn == True:
    lr = 0.00001
    current_model = model_names[1]
    path_cpt_file = f'cpts/{current_model}smnist.cpt'
    model_name = 'GRU RNN'
elif init_lstmrnn == True:
    lr = 0.00001
    current_model = model_names[2]
    path_cpt_file = f'cpts/{current_model}smnist.cpt'
    model_name = 'LSTM RNN'
elif init_hippornn == True:
    lr = 0.00001
    current_model = model_names[3]
    path_cpt_file = f'cpts/{current_model}smnist.cpt'
    model_name = 'HIPPO RNN'

def train (train_loader, model, optimizer, loss_f):
    '''
    Performs the training loop. 
    Input: train loader (torch loader)
           model (torch model)
           optimizer (torch optimizer)
           loss function (torch loss).
    Output: No output.
    '''
    model.train()
    correct = 0 
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # turn [64, 784] to [64, 784, 784]
        x_expanded = x[:, None, ...].expand(x.shape[0], x.shape[1], x.shape[1]).to(device)
        out = model(x_expanded)
        del x
        class_prob = F.softmax(out, dim = 1)
        loss_val = loss_f(class_prob, y)
        del out, class_prob
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        del loss_val    
    return None 

def evaluate (data_loader, model, loss_f):
    '''
    Input: test or train loader (torch loader) 
           model (torch model)
           loss function (torch loss)
    Output: loss (torch float)
            accuracy (torch float)
    '''
    loss_lst = []
    top1_acc_lst = []
    top5_acc_lst = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            x_expanded = x[:, None, ...].expand(x.shape[0], x.shape[1], x.shape[1]).to(device)
            out = model(x_expanded)
            del x, x_expanded
            class_prob = F.softmax(out, dim = 1)
            pred = torch.argmax(class_prob, dim = 1)
            loss_val = loss_f(class_prob, y)
            loss_lst.append(float(loss_val.item()))
            top1_acc_val = top1accuracy(class_prob, y)
            top5_acc_val = top5accuracy(class_prob, y)
            top1_acc_lst.append(float(top1_acc_val))
            top5_acc_lst.append(float(top5_acc_val))
            del y, out  
        # compute average loss
        loss_val = round(sum(loss_lst) / len(loss_lst), 4)
        top1_acc = round(sum(top1_acc_lst) / len(top1_acc_lst),  4)
        top5_acc = round(sum(top5_acc_lst) / len(top5_acc_lst), 4)
        return (loss_val, top1_acc, top5_acc)

def main():
    last_run = 0
    last_epoch = 0
    train_loss_lst = []
    test_loss_lst = []
    train_acc_lst = []
    test_acc_lst = []
    train_top1acc_lst = []
    test_top1acc_lst = []
    train_top5acc_lst = []
    test_top5acc_lst = []

    continue_training = False

    # if we continue training extract last epoch and last run from checkpoint
    if continue_training == True:
        checkpoint = torch.load(path_cpt_file, map_location = device)
        last_epoch = checkpoint['epoch']
        last_run = checkpoint['run']
        print(f"Continue training from run: {last_run + 1} and epoch: {last_epoch + 1}.")
  
        strip_square_brackets(f"results/{current_model}train_loss_run{last_run + 1}.txt")
        strip_square_brackets(f"results/{current_model}train_top1acc_run{last_run + 1}.txt")
        strip_square_brackets(f"results/{current_model}train_top5acc_run{last_run + 1}.txt")
            
        strip_square_brackets(f"results/{current_model}test_loss_run{last_run + 1}.txt")
        strip_square_brackets(f"results/{current_model}test_top1acc_run{last_run + 1}.txt")
        strip_square_brackets(f"results/{current_model}test_top5acc_run{last_run + 1}.txt")

        train_loss_lst = list(genfromtxt(f"results/{current_model}train_loss_run{last_run + 1}.txt", delimiter=','))
        train_top1acc_lst = list(genfromtxt(f"results/{current_model}train_top1acc_run{last_run + 1}.txt", delimiter=','))
        train_top5acc_lst = list(genfromtxt(f"results/{current_model}train_top5acc_run{last_run + 1}.txt", delimiter=','))

        test_loss_lst = list(genfromtxt(f"results/{current_model}test_loss_run{last_run + 1}.txt", delimiter=','))
        test_top1acc_lst = list(genfromtxt(f"results/{current_model}test_top1acc_run{last_run + 1}.txt", delimiter=','))
        test_top5acc_lst = list(genfromtxt(f"results/{current_model}test_top5acc_run{last_run + 1}.txt", delimiter=','))

    for run in range(last_run, nruns):
        # within the run loop if we continue training we initalise model and 
        # optimizer with parameters from the checkpoint 
        if continue_training == True:
            
            if init_simplernn == True: 
                model = SimpleRNN(input_size = input_size*input_size, hidden_size = hidden_size, num_layers = nlayers, output_size = 10, activation = 'relu').to(device)
            
            elif init_grurnn == True:
                 model = GruRNN(input_size = input_size*input_size, hidden_size = hidden_size, num_layers = nlayers, output_size = 10).to(device)
            
            elif init_lstmrnn == True:
                 model = LstmRNN(input_size = input_size*input_size, hidden_size = hidden_size, num_layers = nlayers, output_size = 10).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
            checkpoint = torch.load(path_cpt_file, map_location = device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Run {run + 1}/{nruns}: {model_name} from a previous checkpoint initalised with {nlayers} layers and {hidden_size} number of hidden neurons.')
        
        elif continue_training == False:
            
            if init_simplernn == True: 
                model = SimpleRNN(input_size = input_size*input_size, hidden_size = hidden_size, num_layers = nlayers, output_size = 10, activation = 'relu').to(device)
            
            elif init_grurnn == True:
                model = GruRNN(input_size = input_size*input_size, hidden_size = hidden_size, num_layers = nlayers, output_size = 10).to(device)
            
            elif init_lstmrnn == True:
                 model = LstmRNN(input_size = input_size*input_size, hidden_size = hidden_size, num_layers = nlayers, output_size = 10).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
            print(f'Run {run + 1}/{nruns}: {model_name} initalised with {nlayers} layers and {hidden_size} number of hidden neurons.')

            # ensure that lists are empty when a new model is initalised so
            # that lists from a previous run do not interfere with storage
            train_loss_lst = []
            test_loss_lst = []
            train_acc_lst = []
            test_acc_lst = []
            train_top1acc_lst = []
            test_top1acc_lst = []
            train_top5acc_lst = []
            test_top5acc_lst = []

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = nepochs - warmup, eta_min = 0.00001)
        loss_f = nn.CrossEntropyLoss()

        if mnist == True:
            train_dataset = torchvision.datasets.MNIST(root = data_dir,
                                                train = True, 
                                                transform = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)]),
                                                download = True)

            test_dataset = torchvision.datasets.MNIST(root =  data_dir,
                                                train = False, 
                                                transform = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)]))
        if cifar10 == True:
            train_dataset = torchvision.datasets.CIFAR10(root = './data', train=True,
                                                download = True, transform = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)]))

            testset = torchvision.datasets.CIFAR10(root = './data', train=False,
                                            download = True, transform = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)]))

        # we drop the last batch to ensure each batch has the same size
        train_loader = DataLoader(dataset = train_dataset,
                                            batch_size = batch_size, 
                                            shuffle = True, drop_last = True)
        
        test_loader = DataLoader(dataset = test_dataset,
                                            batch_size = batch_size, 
                                            shuffle = False, drop_last = True)

        for epoch in range(last_epoch, nepochs):
            # adjust learning rate
            # 1. linear increase from 0.00001 to 0.0001 over 5 epochs
            if epoch + last_epoch > 0 and epoch + last_epoch <= warmup:
                optimizer.param_groups[0]['lr'] =  0.00001 + (0.00009/warmup) * (epoch + last_epoch)
            # 2. decrease from 0.0001 to 0 using cosine annealing 
            # calling scheduler step before optimizer step will trigger a warning
            # however since we adjust learning rate based on epoch and then want
            # to train this warning can be ignored.
            elif epoch + last_epoch > warmup:
                scheduler.step()

            # train 
            train(train_loader, model, optimizer, loss_f)
            train_loss_value, train_top1acc_value, train_top5acc_value = evaluate(train_loader, model, loss_f)
            train_loss_lst.append(train_loss_value)
            train_top1acc_lst.append(train_top1acc_value)
            train_top5acc_lst.append(train_top5acc_value)
            
            # test 
            test_loss_value, test_top1acc_value, test_top5acc_value = evaluate(test_loader, model, loss_f)
            test_loss_lst.append(test_loss_value)
            test_top1acc_lst.append(test_top1acc_value)
            test_top5acc_lst.append(test_top5acc_value)
            
            print(f"Epoch:{epoch + 1}   Train[Loss:{train_loss_value} Top1 Acc:{train_top1acc_value}  Top5 Acc:{train_top5acc_value}]")
            print(f"Epoch:{epoch + 1}   Test[Loss:{test_loss_value}   Top1 Acc:{test_top1acc_value}   Top5 Acc:{test_top5acc_value}]")

            if save_model == True and ((epoch + 1) % 3) == 0 or epoch == nepochs - 1:
                torch.save({
                    'epoch': epoch,
                    'run': run,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, path_cpt_file)
                print(f"Checkpoint and evaluation at epoch {epoch + 1} stored")
                with open(f'results/{current_model}train_loss_run{run + 1}.txt','w') as values:
                    values.write(str(train_loss_lst))
                with open(f'results/{current_model}train_top1acc_run{run + 1}.txt','w') as values:
                    values.write(str(train_top1acc_lst))
                with open(f'results/{current_model}train_top5acc_run{run + 1}.txt','w') as values:
                    values.write(str(train_top5acc_lst))

                with open(f'results/{current_model}test_loss_run{run + 1}.txt','w') as values:
                    values.write(str(test_loss_lst))
                with open(f'results/{current_model}test_top1acc_run{run + 1}.txt','w') as values:
                    values.write(str(test_top1acc_lst))
                with open(f'results/{current_model}test_top5acc_run{run + 1}.txt','w') as values:
                    values.write(str(test_top5acc_lst))
            
            # if epoch has reached last epoch reset last_epoch variable to zero
            # to ensure that once we start another run we start at the first epoch
            # and not an epoch we held onto from continuing training
            if epoch == nepochs - 1:
                last_epoch = 0
                continue_training = False

if __name__ == "__main__":
    main()