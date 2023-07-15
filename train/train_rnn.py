import torch
from torch.utils.data.dataset import T_co
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from cells.rnncells import RnnCell, GruCell, LstmCell
from models.rnn import SimpleRNN, GruRNN, LstmRNN
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import top1accuracy, top5accuracy, strip_square_brackets
from numpy import genfromtxt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# defines number of runs or repititions of the experiment
nruns = 5

#continue_training = True

data_dir =  'data/'

model_names = ['simple', 
                'gru',
                'lstm',
                'hippo']

dataset_names = ['smnist',
                'mnist'
                'scifar']

floatTypes = [
  "weight_decay", "lr"
]

intTypes = [
  "sequence_length", 
  "input_size",
  "hidden_size",
  "nlayers",
  "nclasses",
  "batch_size",
  "nepochs",
  "nruns",
  "warmup",
]

boolTypes = [
  "save_model",
  "continue_training"
]

def initialize_with_args(_arguments):
  arguments = {
      "model_name": "simple", 
      "dataset_name": "smnist", 
      "save_model": True,
      "continue_training": False,
      "weight_decay": 0.0005,
      "sequence_length": 28 ,
      "input_size": 28,
      "hidden_size": 64,
      "nlayers": 2,
      "nclasses": 10,
      "batch_size": 64,
      "nepochs": 100,
      "nruns": 5,
      "warmup": 5,
      "init_simplernn": False,
      "init_grurnn": False,
      "init_lstmrnn": False,
      "init_hippornn": False,
      "mnist": False,
      "cifar10": False,
      "current_model": "",
      "lr": -1,
      "path_cpt_file": ""
      "model_name",
  }
  arguments = {**arguments, **_arguments}

  for key, value in arguments.items():
    if key in floatTypes:
      arguments[key] = float(value)
    if key in intTypes:
      arguments[key] = int(value)
    if key in boolTypes:
      arguments[key] = bool(value)

  if arguments["model_name"] not in model_names:
    print(f"model name {arguments['model_name']} was not found, use simple, gru, lstm or hippo")
    return
  if arguments["dataset_name"] not in dataset_names:
    print(f"dataset name {arguments['_dataset_name']} was not found")
    return

  match arguments["model_name"]:
    case "simple":
      arguments["init_simplernn"] = True
      arguments["lr"] = 0.00001
      arguments["current_model"] = model_names[0]
      arguments["path_cpt_file"] = f'cpts/{arguments["current_model"]}rnn_smnist.cpt'
      arguments["model_name"] = 'Simple Rnn'
    case "gru":
      arguments["init_grurnn"] = True
      arguments["lr"] = 0.00001
      arguments["current_model"]= model_names[1]
      arguments["path_cpt_file"]= f'cpts/{arguments["current_model"]}rnn_smnist.cpt'
      arguments["model_name"]= 'Gru Rnn'
    case "lstm":
      arguments["init_lstmrnn"] = True
      arguments["lr"]= 0.00001
      arguments["current_model"]= model_names[2]
      arguments["path_cpt_file"]= f'cpts/{arguments["current_model"]}rnn_smnist.cpt'
      arguments["model_name"]= 'Lstm Rnn'
    case "hippo":
      arguments["init_hippornn"] = True
      arguments["lr"]= 0.00001
      arguments["current_model"]= model_names[3]
      arguments["path_cpt_file"]= f'cpts/{arguments["current_model"]}hippo_smnist.cpt'
      arguments["model_name"]= 'Simple Hippo'

  match arguments["dataset_name"]:
    case "smnist":
      arguments["mnist"] = True
    case "mnist":
      arguments["mnist"] = True
    case "cifar10":
      arguments["cifar10"] = True

  main(arguments)

def train(train_loader, model, optimizer, loss_f):
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
        x_expanded = x.unsqueeze(-1) # x[:, None, ...].expand(x.shape[0], x.shape[1], x.shape[1]).to(device)
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
            x_expanded = x.unsqueeze(-1) #x[:, None, ...].expand(x.shape[0], x.shape[1], x.shape[1]).to(device)
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

def main(arguments):
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

    continue_training = arguments["continue_training"]
    print(continue_training)
    # if we continue training extract last epoch and last run from checkpoint
    if continue_training == True:
        checkpoint = torch.load(arguments["path_cpt_file"], map_location = device)
        last_epoch = checkpoint['epoch']
        last_run = checkpoint['run']
        print(f"Continue training from run: {last_run + 1} and epoch: {last_epoch + 1}.")
  
        strip_square_brackets(f"results/{arguments['current_model']}rnn_train_loss_run{last_run + 1}.txt")
        strip_square_brackets(f"results/{arguments['current_model']}rnn_train_top1acc_run{last_run + 1}.txt")
        strip_square_brackets(f"results/{arguments['current_model']}rnn_train_top5acc_run{last_run + 1}.txt")
            
        strip_square_brackets(f"results/{arguments['current_model']}rnn_test_loss_run{last_run + 1}.txt")
        strip_square_brackets(f"results/{arguments['current_model']}rnn_test_top1acc_run{last_run + 1}.txt")
        strip_square_brackets(f"results/{arguments['current_model']}rnn_test_top5acc_run{last_run + 1}.txt")

        train_loss_lst = list(genfromtxt(f"results/{arguments['current_model']}rnn_train_loss_run{last_run + 1}.txt", delimiter=','))
        train_top1acc_lst = list(genfromtxt(f"results/{arguments['current_model']}rnn_train_top1acc_run{last_run + 1}.txt", delimiter=','))
        train_top5acc_lst = list(genfromtxt(f"results/{arguments['current_model']}rnn_train_top5acc_run{last_run + 1}.txt", delimiter=','))

        test_loss_lst = list(genfromtxt(f"results/{arguments['current_model']}rnn_test_loss_run{last_run + 1}.txt", delimiter=','))
        test_top1acc_lst = list(genfromtxt(f"results/{arguments['current_model']}rnn_test_top1acc_run{last_run + 1}.txt", delimiter=','))
        test_top5acc_lst = list(genfromtxt(f"results/{arguments['current_model']}rnn_test_top5acc_run{last_run + 1}.txt", delimiter=','))

    for run in range(last_run, nruns):
        # within the run loop if we continue training we initalise model and 
        # optimizer with parameters from the checkpoint 
        if continue_training == True:
            
            if arguments["init_simplernn"] == True: 
                model = SimpleRNN(input_size = arguments['input_size']**2, hidden_size = arguments["hidden_size"], num_layers = arguments["nlayers"], output_size = 10, activation = 'relu').to(device)
            
            elif arguments["init_grurnn"] == True:
                 model = GruRNN(input_size = arguments['input_size']**2, hidden_size = arguments["hidden_size"], num_layers = arguments["nlayers"], output_size = 10).to(device)
            
            elif arguments["init_lstmrnn"] == True:
                 model = LstmRNN(input_size = arguments['input_size']**2, hidden_size = arguments["hidden_size"], num_layers = arguments["nlayers"], output_size = 10).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr = arguments["lr"], weight_decay = arguments["weight_decay"])
            checkpoint = torch.load(arguments["path_cpt_file"], map_location = device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Run {run + 1}/{nruns}: {arguments["model_name"]} from a previous checkpoint initalised with {arguments["nlayers"]} layers and {arguments["hidden_size"]} number of hidden neurons.')
        
        elif continue_training == False:
            
            if arguments["init_simplernn"] == True: 
                model = SimpleRNN(input_size = arguments['input_size']**2, hidden_size = arguments["hidden_size"], num_layers = arguments["nlayers"], output_size = 10, activation = 'relu').to(device)
            
            elif arguments["init_grurnn"] == True:
                model = GruRNN(input_size = arguments['input_size']**2, hidden_size = arguments["hidden_size"], num_layers = arguments["nlayers"], output_size = 10).to(device)
            
            elif arguments["init_lstmrnn"] == True:
                 model = LstmRNN(input_size = arguments['input_size']**2, hidden_size = arguments["hidden_size"], num_layers = arguments["nlayers"], output_size = 10).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr = arguments["lr"], weight_decay = arguments["weight_decay"])
            print(f'Run {run + 1}/{nruns}: {arguments["model_name"]} initalised with {arguments["nlayers"]} layers and {arguments["hidden_size"]} number of hidden neurons.')

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

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = arguments["nepochs"] - arguments["warmup"], eta_min = 0.00001)
        loss_f = nn.CrossEntropyLoss()

        if arguments["mnist"] == True:
            train_dataset = torchvision.datasets.MNIST(root = data_dir,
                                                train = True, 
                                                transform = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)]),
                                                download = True)

            test_dataset = torchvision.datasets.MNIST(root =  data_dir,
                                                train = False, 
                                                transform = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)]))
        if arguments["cifar10"] == True:
            train_dataset = torchvision.datasets.CIFAR10(root = './data', train=True,
                                                download = True, transform = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)]))

            testset = torchvision.datasets.CIFAR10(root = './data', train=False,
                                            download = True, transform = T.Compose([T.ToTensor(), T.Lambda(torch.flatten)]))

        # we drop the last batch to ensure each batch has the same size
        train_loader = DataLoader(dataset = train_dataset,
                                            batch_size = arguments["batch_size"], 
                                            shuffle = True, drop_last = True)
        
        test_loader = DataLoader(dataset = test_dataset,
                                            batch_size = arguments["batch_size"], 
                                            shuffle = False, drop_last = True)

        for epoch in range(last_epoch, arguments["nepochs"]):
            # adjust learning rate
            # 1. linear increase from 0.00001 to 0.0001 over 5 epochs
            if epoch + last_epoch > 0 and epoch + last_epoch <= arguments["warmup"]:
                optimizer.param_groups[0]['lr'] =  0.00001 + (0.00009/arguments["warmup"]) * (epoch + last_epoch)
            # 2. decrease from 0.0001 to 0 using cosine annealing 
            # calling scheduler step before optimizer step will trigger a warning
            # however since we adjust learning rate based on epoch and then want
            # to train this warning can be ignored.
            elif epoch + last_epoch > arguments["warmup"]:
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

            if arguments["save_model"] == True and ((epoch + 1) % 3) == 0 or epoch == arguments["nepochs"] - 1:
                torch.save({
                    'epoch': epoch,
                    'run': run,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, arguments["path_cpt_file"])
                print(f"Checkpoint and evaluation at epoch {epoch + 1} stored")
                with open(f'results/{arguments["current_model"]}rnn_train_loss_run{run + 1}.txt','w') as values:
                    values.write(str(train_loss_lst))
                with open(f'results/{arguments["current_model"]}rnn_train_top1acc_run{run + 1}.txt','w') as values:
                    values.write(str(train_top1acc_lst))
                with open(f'results/{arguments["current_model"]}rnn_train_top5acc_run{run + 1}.txt','w') as values:
                    values.write(str(train_top5acc_lst))

                with open(f'results/{arguments["current_model"]}rnn_test_loss_run{run + 1}.txt','w') as values:
                    values.write(str(test_loss_lst))
                with open(f'results/{arguments["current_model"]}rnn_test_top1acc_run{run + 1}.txt','w') as values:
                    values.write(str(test_top1acc_lst))
                with open(f'results/{arguments["current_model"]}rnn_test_top5acc_run{run + 1}.txt','w') as values:
                    values.write(str(test_top5acc_lst))
            
            # if epoch has reached last epoch reset last_epoch variable to zero
            # to ensure that once we start another run we start at the first epoch
            # and not an epoch we held onto from continuing training
            if epoch == arguments["nepochs"] - 1:
                last_epoch = 0
                continue_training = False

# if __name__ == "__main__":
#     main()