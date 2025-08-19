from turtle import forward
import numpy as np
from rsl_rl.modules.actor_critic import get_activation

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU
from torch.nn.utils.parametrizations import spectral_norm

class Estimator_ori(nn.Module):
    def __init__(self,  input_dim,
                        output_dim,
                        hidden_dims=[256, 128, 64],
                        activation="elu",
                        **kwargs):
        super(Estimator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        activation = get_activation(activation)
        estimator_layers = []
        estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                estimator_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                estimator_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator_layers)
    
    def forward(self, input):
        return self.estimator(input)
    
    def inference(self, input):
        with torch.no_grad():
            return self.estimator(input)


class Estimator(nn.Module):
    def __init__(self, input_dim,  # 132
                 output_dim,  # 8
                 gru_hidden_dim=256,
                 gru_layers=1,
                 hidden_dims=[256, 128, 64],
                 activation="elu",
                 **kwargs):
        super(Estimator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_layers = gru_layers

        activation = get_activation(activation)
        estimator_layers = []
        estimator_layers.append(nn.Linear(self.gru_hidden_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                estimator_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                estimator_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator_layers)

        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_hidden_dim, num_layers=gru_layers, batch_first=True)

        self.attn_fc = nn.Linear(self.gru_hidden_dim, 1)

    def forward(self, input):
        gru_out, _ = self.gru(input)
        # last_hidden = gru_out[:, -1, :] # 池化
        attn_scores = self.attn_fc(gru_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)
        return self.estimator(context)

    def inference(self, input):
        with torch.no_grad():
            gru_out, _ = self.gru(input)
            # last_hidden = gru_out[:, -1, :] # 池化
            attn_scores = self.attn_fc(gru_out).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)
            return self.estimator(context)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
class Discriminator(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(nn.Linear(n_states, hidden_dims[0]))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(nn.Linear(hidden_dims[l], n_skills))
            else:
                discriminator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)
        # self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        # init_weight(self.hidden1)
        # self.hidden1.bias.data.zero_()
        # self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        # init_weight(self.hidden2)
        # self.hidden2.bias.data.zero_()
        # self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        # init_weight(self.q, initializer="xavier uniform")
        # self.q.bias.data.zero_()

    def forward(self, states):
        return self.discriminator(states)

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)

class DiscriminatorLSD(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(DiscriminatorLSD, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(spectral_norm(nn.Linear(n_states, hidden_dims[0])))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(spectral_norm(nn.Linear(hidden_dims[l], n_skills)))
            else:
                discriminator_layers.append(spectral_norm(nn.Linear(hidden_dims[l], hidden_dims[l + 1])))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)
        

    def forward(self, states):
        return self.discriminator(states)

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)
        
class DiscriminatorContDIAYN(nn.Module):
    def __init__(self, n_states, 
                 n_skills, 
                 hidden_dims=[256, 128, 64], 
                 activation="elu"):
        super(DiscriminatorContDIAYN, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills

        activation = get_activation(activation)
        discriminator_layers = []
        discriminator_layers.append(nn.Linear(n_states, hidden_dims[0]))
        discriminator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                discriminator_layers.append(nn.Linear(hidden_dims[l], n_skills))
            else:
                discriminator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                discriminator_layers.append(activation)
        self.discriminator = nn.Sequential(*discriminator_layers)

    def forward(self, states):
        return torch.nn.functional.normalize(self.discriminator(states))

    def inference(self, states):
        with torch.no_grad():
            return self.discriminator(states)