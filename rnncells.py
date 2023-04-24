#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 22:52:13 2022

@author: sen
"""
import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
from torch.autograd import Variable

class RnnCell(nn.Module):
    def __init__(self, input_size, hidden_size, activation="tanh"):
        super(RnnCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        if self.activation not in ["tanh", "relu", "sigmoid"]:
            raise ValueError("Invalid nonlinearity selected for RNN. Please use tanh, relu or sigmoid.")

        self.input2hidden = nn.Linear(input_size, hidden_size)
        # hidden2hidden when we have more than 1 RNN stacked
        # hidden2out when we have only 1 RNN
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)
        
        self.init_weights_normal()
        
    def forward(self, input, hidden_state = None):
        '''
        Inputs: input (torch tensor) of shape [batchsize, input_size]
                hidden state (torch tensor) of shape [batchsize, hiddensize]
        Output: output (torch tensor) of shape [batchsize, hiddensize ]
        '''

        # initalise hidden state at first iteration so if none
        if hidden_state is None:
            hidden_state = torch.zeros(input.shape[0], self.hidden_size).to(device)

        # here the rnn magic happens, once we have a hidden state it becomes the
        # input for the next hidden state, that way we keep an internal memory
        hidden_state = (self.input2hidden(input) + self.hidden2hidden(hidden_state))

        # takes output from hidden and apply activation
        if self.activation == "tanh":
            out = torch.tanh(hidden_state)
        elif self.activation == "relu":
            out = torch.relu(hidden_state)
        elif self.activation == "sigmoid":
            out = torch.sigmoid(hidden_state) 
        return out

    def init_weights_normal(self):
      # iterate over parameters or weights theta
      # and initalise them with a normal centered at 0 with 0.02 spread.
      for weight in self.parameters():
          weight.data.normal_(0, 0.02)

class GruCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GruCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)

        self.init_weights_normal()

    def init_weights_normal(self):
      # iterate over parameters or weights theta
      # and initalise them with a normal centered at 0 with 0.02 spread.
      for weight in self.parameters():
          weight.data.normal_(0, 0.02)

    def forward(self, input, hidden_state = None):
        '''
        Inputs: input (torch tensor) of shape [batchsize, input_size]
                hidden state (torch tensor) of shape [batchsize, hiddensize]
        Output: output (torch tensor) of shape [batchsize, hiddensize]
        '''

        if hidden_state is None:
            hidden_state = torch.zeros(input.shape[0], self.hidden_size).to(device)

        input_reset = self.input2hidden(input)
        input_update = self.input2hidden(input)
        input_new = self.input2hidden(input)
        hidden_reset = self.hidden2hidden(hidden_state)
        hidden_update = self.hidden2hidden(hidden_state)
        hidden_new = self.hidden2hidden(hidden_state)

        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        new_gate = torch.tanh(input_new + (reset_gate * hidden_new))

        out = update_gate * hidden_state + (1 - update_gate) * new_gate

        return out

class LstmCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LstmCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input2hidden = nn.Linear(input_size, hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)
        self.init_weights_normal()
    
    def init_weights_normal(self):
        # iterate over parameters or weights theta
        # and initalise them with a normal centered at 0 with 0.02 spread.
        for weight in self.parameters():
            weight.data.normal_(0, 0.02)

    def forward(self, input, hidden_state = None):
        '''
        Inputs: input (torch tensor) of shape [batchsize, input_size]
                hidden state (torch tensor) of shape [batchsize, hiddensize]
        Output: output (torch tensor) of shape [batchsize, hiddensize]
        '''
        if hidden_state is None:
            hidden_state = torch.zeros(input.shape[0], self.hidden_size).to(device)
            hidden_state = (hidden_state, hidden_state)

        hidden_state, previous_cell_state = hidden_state

        input_gate = self.input2hidden(input) + self.hidden2hidden(hidden_state)
        forget_gate = self.input2hidden(input) + self.hidden2hidden(hidden_state)
        cell_gate = self.input2hidden(input) + self.hidden2hidden(hidden_state)
        output_gate = self.input2hidden(input) + self.hidden2hidden(hidden_state)

        input_gate_activation = torch.sigmoid(input_gate)
        forget_gate_activation = torch.sigmoid(forget_gate)
        cell_gate_activation = torch.tanh(cell_gate)
        output_gate_activation = torch.sigmoid(output_gate)

        updated_cell_state = previous_cell_state * forget_gate_activation + input_gate_activation * cell_gate_activation

        # output for the hidden
        out = output_gate_activation * torch.tanh(updated_cell_state)

        return (out, updated_cell_state)
