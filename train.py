import torch
import torch.nn as nn
import numpy as np
from preprocess import xor_data_generate



def sample_action(action, n, batch):
    """
    among m data points select the n most probable data points for training
    :param action: dim (m, 1), each entry is the probability that we select this data point
    :param n: number of data points we want to sample from m data points
    :param batch: batch of m data points fot sampling, dim: (m, d0), where d0 is the dim of each x_i
    :return: sampled_data, dim (n, d0); sampled_idx, dim (n, 1)

    """

    sampled_idx = torch.topk(action, n)[1]
    sampled_data = batch[sampled_idx]


    return sampled_data, sampled_idx


def sample(X, Y, n):
    """
    function that samples n data points among m evaluation data points uniformly
    """
    m = X.shape[0]
    sampled_idx = np.random.choice(np.arange(0, m), n, replace=False)
    sampled_X = X[sampled_idx]
    sampled_Y = Y[sampled_idx]

    return sampled_X, sampled_Y


def reward_computing(encoder, classify_nn, X_eval, Y_eval, loss_last, reward_amplify, passive_drive):
    """
    compute and return the reward R(t) at time step t, which is defined as the difference between two cross-entropy loss
    in eval set on time step t and (t-1)
    :param encoder: (n, d0) --> (n, h)
    :param classify_nn: (n, h) --> (n, C)
    :param X_eval: dim (n, d0)
    :param Y_eval: dim (n, 1)
    :param loss_last: torch.Variable
    :return: reward: torch.Variable
    """
    Y_predict = classify_nn(encoder(X_eval))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(Y_predict, Y_eval.view(-1))

    reward = (loss_last - loss) * reward_amplify
    reward.detach_()

    return reward, loss


def choose_action(V):
    """
    given a batch of state-action value pairs, determine which data points should be chosen
    by comparing Q(s,a0) and Q(s,a1)
    :param V: a pair of tensors, each tensor contains m state-action values: dim(V[0]) = dim(V[1]) = (m,1)
    :return:
    """

    m = V[0].shape[0]
    print("V[0]:", V[0])
    print("V[1]:", V[1])
    idx = [i for i in range(m) if V[1][i][0] > V[0][i][0]]
    print("idx:", idx)
    return idx


def to_one_hot(action, m):
    one_hot_action = [0] * m
    for i in action:
        one_hot_action[i] = 1
    return one_hot_action


def train_step(E, Q, Q_t, memory, X, Y, C, optimizer_C, optimizer_E, optimizer_Q, eval_X, eval_Y, loss_last, gamma,
               reward_amplify, passive_drive):
    """
    train process for each step, update Q-network and classification network C (and encoder E)
    :param E:
    :param Q:
    :param Q_t:
    :param memory:
    :param S:
    :param Y:
    :param C:
    :param optimizer_C:
    :param optimizer_Q:
    :param train_X:
    :param train_Y:
    :param eval_X:
    :param eval_Y:
    :param sample_size:
    :param loss_last:
    :param gamma:
    :param reward_amplify:
    :param passive_drive:
    :return:
    """

    # epsilon-greedy policy
    S = E(X)
    m = S.shape[0]
    action = []
    while len(action) == 0:
        rand1 = np.random.rand()
        if rand1 < Q.epsilon:
            # with probability epsilon, randomly select which data points are used (1/2-1/2 probability)
            n, p = 1, 0.5
            br = np.random.binomial(n, p, m)
            action = [i for i in range(m) if br[i] == 1]
        else:
            # with probability (1-epsilon), determine which data points are selected by computing their Q-values
            V = Q(S)    # m pairs of (Q(s, a0), Q(s,a1)) state-action values
            action = choose_action(V)

    # execute action (use selected data to train classifier C)
    S_sampled = S[action]
    Y_sampled = Y[action].view(-1)
    #print("length of S_sampled:", S_sampled.shape[0])
    T = C(S_sampled)
    criterion_C = nn.NLLLoss()
    #print("shape of T:", T.shape)
    #print("shape of Y_sampled:", Y_sampled.shape)
    C_loss = criterion_C(T, Y_sampled)
    C_loss.backward(retain_graph=True)
    optimizer_C.step()
    optimizer_E.step()

    # get the step reward
    reward, loss = reward_computing(E, C, eval_X, eval_Y, loss_last, reward_amplify, passive_drive=passive_drive)

    # sample from training set to get the next batch of data,
    sampled_X, sampled_Y = xor_data_generate(m)
    sampled_X = sampled_X.cuda()
    sampled_Y = sampled_Y.cuda()

    # encode it to obtain s_(t+1)
    S_new = E(sampled_X)

    # store the m transition tuples into memory, using average reward reshaping:
    action = to_one_hot(action, m)
    for i in range(m):
        transition = [S[i], action[i], reward/m, S_new[i]]
        memory.add_transition(transition)

    # sample a random transition tuple from memory
    sampled_transition = memory.sampling()

    # perform temporal difference learning, compute y_j
    s_j = sampled_transition[0]
    a_j = sampled_transition[1]     # either 0 or 1
    r_j = sampled_transition[2]
    s_jp1 = sampled_transition[3]

    q0, q1 = Q_t(s_jp1)
    if q0.data > q1.data:
        y_j = r_j + gamma * q0  # y_j: torch.Variable
    else:
        y_j = r_j + gamma * q1

    q0, q1 = Q(s_j)
    criterion_Q = nn.MSELoss()
    y_j = y_j.detach()
    if a_j == 0:
        Q_loss = criterion_Q(q0, y_j)
    else:
        Q_loss = criterion_Q(q1, y_j)

    Q_loss.backward(retain_graph=True)
    optimizer_Q.step()

    return sampled_X, sampled_Y, loss, reward



























