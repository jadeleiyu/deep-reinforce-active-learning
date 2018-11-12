import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Simple_Q_Net(nn.Module):
    def __init__(self, config):
        """
        a simple neural Q-learning network model that take state representation pair as input and outputs Q(s,a) value
        for a0 and a1.
        this network only uses several linear layers instead of RNNs for NLP tasks
        :param state_dim: Dimension of input state (int), which equals the dim of the output of shared encoder network
        :return
        """

        super(Simple_Q_Net, self).__init__()

        self.state_dim = config.state_dim
        self.epsilon = config.epsilon

        self.fcs1 = nn.Linear(self.state_dim, 10)
        #self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(10, 20)
        #self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        self.fcq0 = nn.Linear(20, 1)
        self.fcq1 = nn.Linear(20, 1)


    def forward(self, state):
        """
        compute state-value function Q(s,a0) and Q(s,a1) obtained from critic network,
        where a0 is "do not select", a1 is "select"
        :param state: Input state (Torch Variable : [m,state_dim] )
        :return: a pair of state-value function  (Q(S,a0),  Q(S,a1)) each is a torch.Variable of dim (m, 1)
        """
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))

        q0 = self.fcq0(s2)
        q1 = self.fcq1(s2)

        return q0, q1   # to compare value, use q0[0] < q1[0] or q0.data < q1.data


class Shared_Encoder(nn.Module):
    """
    encoder shared by the neural classifier and the Q-network
    """

    def __init__(self, config):

        super(Shared_Encoder, self).__init__()

        self.input_dim = config.input_dim
        self.hidden = config.hidden

        self.fc1 = nn.Linear(self.input_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)

    def forward(self, X):
        """

        :param X: input data batch, dim: (m, input_dim), where m is the batch size and also the dim of the state vector
        :return: encoded representation of data batch, dim: (m, h)
        """

        h1 = F.relu(self.fc1(X))
        h2 = F.relu(self.fc2(h1))

        return h2.view(-1, self.hidden)


class CCNN(nn.Module):
    """
    cascade-correlation neural network, to be implemented in pytorch
    """


class SimpleNNClassifier(nn.Module):

    def __init__(self, config):

        super(SimpleNNClassifier, self).__init__()

        self.state_dim = config.hidden
        self.hidden = config.hidden
        self.output_dim = config.output_dim

        self.fc1 = nn.Linear(self.state_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.output_dim)

    def forward(self, s):
        """

        :param s: state vector of dim (m, h), where m is the size of the data batch
        :return: a batch of class vectors, dim: (m, C), where C is the output_dim (# of all possible classes)
        """

        h = F.relu(self.fc1(s))
        z = F.softmax(self.fc2(h))

        return z.view(-1, self.output_dim)


class Memory:

    def __init__(self, N):
        self.capacity = N
        self.transition_list = []

    def add_transition(self, transition):
        # the memory always keeps the last N transition experiences
        if len(self.transition_list) < self.capacity:
            self.transition_list.append(transition)
        else:
            self.transition_list = self.transition_list[1:]
            self.transition_list.append(transition)

    def sampling(self):
        assert len(self.transition_list) > 0
        l = len(self.transition_list)
        #print("length of transition list:", l)
        rand_idx = random.randint(0, l-1)
        #print("rand_idx:", rand_idx)
        return self.transition_list[rand_idx]
















