#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 22:52:13 2022

@author: sen
"""
import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from rnncells import RnnCell, GruCell, LstmCell
import numpy as np
from torch.autograd import Variable

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, activation='relu'):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        if activation == 'tanh':
            self.rnn_cell_list.append(RnnCell(self.input_size, self.hidden_size, "tanh"))
            
            for l in range(1, self.num_layers):
                
                self.rnn_cell_list.append(RnnCell(self.hidden_size, self.hidden_size, "tanh"))

        elif activation == 'relu':
            self.rnn_cell_list.append(RnnCell(self.input_size, self.hidden_size, "relu"))
            
            for l in range(1, self.num_layers):
                
                self.rnn_cell_list.append(RnnCell(self.hidden_size, self.hidden_size, "relu"))

        elif activation == 'sigmoid':
            self.rnn_cell_list.append(RnnCell(self.input_size, self.hidden_size, "sigmoid"))
            
            for l in range(1, self.num_layers):
                
                self.rnn_cell_list.append(RnnCell(self.hidden_size, self.hidden_size, "sigmoid"))
        
        else:
            raise ValueError("Invalid activation. Please use tanh, relu or sigmoid activation.")

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden_state=None):
        '''
        Inputs: input (torch tensor) of shape [batchsize, seqence length, inputsize]
        Output: output (torch tensor) of shape [batchsize, outputsize]
        '''

        # initalise hidden state at first timestep so if none
        if hidden_state is None:
            hidden_state = torch.zeros(self.num_layers, input.shape[0], self.hidden_size).to(device)

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            
            hidden.append(hidden_state[layer, :, :])
        
        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1], hidden[layer])
                hidden[layer] = hidden_l

            outs.append(hidden_l)

        # select last time step indexed at [-1]
        out = outs[-1].squeeze()
        out = self.fc(out)
        return out

class GruRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GruRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(GruCell(self.input_size, self.hidden_size))
        
        for l in range(1, self.num_layers):
            
            self.rnn_cell_list.append(GruCell(self.hidden_size, self.hidden_size))
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden_state = None):
        '''
        Inputs: input (torch tensor) of shape [batchsize, seqence length, inputsize]
        Output: output (torch tensor) of shape [batchsize, outputsize]
        '''

        if hidden_state is None:
            hidden_state = torch.zeros(self.num_layers, input.shape[0], self.hidden_size).to(device)

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append(hidden_state[layer, :, :])

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], hidden[layer])
                
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l

            outs.append(hidden_l)

        # select last time step indexed at [-1]
        out = outs[-1].squeeze()
        out = self.fc(out)
        return out

class LstmRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LstmRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(LstmCell(self.input_size, self.hidden_size,))
        
        for l in range(1, self.num_layers):
            
            self.rnn_cell_list.append(LstmCell(self.hidden_size, self.hidden_size,))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden_state = None):
        '''
        Inputs: input (torch tensor) of shape [batchsize, seqence length, inputsize]
        Output: output (torch tensor) of shape [batchsize, outputsize]
        '''

        if hidden_state is None:
            hidden_state = torch.zeros(self.num_layers, input.shape[0], self.hidden_size).to(device)

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((hidden_state[layer, :, :], hidden_state[layer, :, :]))

        for t in range(input.size(1)):

            for layer in range(self.num_layers):

                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](input[:, t, :], (hidden[layer][0], hidden[layer][1]))

                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1][0], (hidden[layer][0], hidden[layer][1]))

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        out = outs[-1].squeeze()

        out = self.fc(out)

        return out

def test_rnn():
  # batch size, sequence length, input size
    model = SimpleRNN(input_size=28*28, hidden_size=128, num_layers=3, output_size=10)
    model = model.to(device)
    x = torch.randn(64, 28*28)
    x = x.unsqueeze(-1)
    vals = torch.ones(64, 28*28, 28*28-1) * (28*28)
    x = torch.cat([x, vals], dim=-1).to(device)
    out = model(x)
    xshape = out.shape
    return x, xshape

def test_gru():
  # batch size, sequence length, input size
    model = GruRNN(input_size=28*28, hidden_size=128, num_layers=3, output_size=10)
    model = model.to(device)
    x = torch.randn(64, 28*28)
    x = x.unsqueeze(-1)
    vals = torch.ones(64, 28*28, 28*28-1) * (28*28)
    x = torch.cat([x, vals], dim=-1).to(device)
    out = model(x)
    xshape = out.shape
    return x, xshape


def test_lstm():
  # batch size, sequence length, input size
    model = LstmRNN(input_size=28*28, hidden_size=128, num_layers=3, output_size=10)
    model = model.to(device)
    x = torch.randn(64, 28*28)
    x = x.unsqueeze(-1)
    vals = torch.ones(64, 28*28, 28*28-1) * (28*28)
    x = torch.cat([x, vals], dim=-1).to(device)
    out = model(x)
    xshape = out.shape
    return x, xshape

testx, xdims = test_rnn()
print("Simple RNN size test: passed.")

testx, xdims = test_gru()
print("Gru RNN size test: passed.")\

testx, xdims = test_gru()
print("LSTM RNN size test: passed.")

