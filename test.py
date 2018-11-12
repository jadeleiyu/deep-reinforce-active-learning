import matplotlib
matplotlib.use('agg')
import torch
import torch.nn as nn
from preprocess import xor_data_generate, twospirals
import models
from utils import get_args, Config
from matplotlib import pyplot as plt
import numpy as np
import math


class Simple_Pair_Net(nn.Module):
    def __init__(self, x_dim):
        super(Simple_Pair_Net, self).__init__()

        self.x_dim = x_dim

        self.fc1 = nn.Linear(x_dim, 3)
        self.fc2 = nn.Linear(x_dim, 3)

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(x)

        return y1, y2


def test_C(C, E):
    X, Y = xor_data_generate(1000)
    X = X.cuda()
    Y = Y.cuda()
    T = C(E(X))

    for i in range(1000):
        print("Y_i:", Y[i])
        print("T_i:", T[i])
    criterion = nn.NLLLoss()
    loss = criterion(T, Y.view(-1))
    print("validation loss of C:", loss.detach().cpu().numpy())


def simple_train_C():
    torch.manual_seed(233)
    torch.cuda.set_device(0)
    args = get_args()
    print("generating config")
    config = Config(
        state_dim=args.hidden,
        input_dim=args.input_dim,
        hidden=args.hidden,
        output_dim=args.num_classes,
        epsilon=args.epsilon
    )
    gamma = args.gamma
    memory = models.Memory(args.capacity)
    m = args.batch_size
    print("initializing networks")
    E = models.Shared_Encoder(config)
    C = models.SimpleNNClassifier(config)

    optimizer_E = torch.optim.Adam(E.parameters(), lr=args.lr, betas=(0., 0.999))
    optimizer_C = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(0., 0.999))

    E.cuda()
    C.cuda()

    X, Y = xor_data_generate(30000)
    X = X.cuda()
    Y = Y.cuda()
    for i in range(30000):
        x = X[i]
        y = Y[i]
        t = C(E(x))

        criterion = nn.NLLLoss()
        loss = criterion(t, y.view(-1))

        loss.backward()

        optimizer_E.step()
        optimizer_C.step()

        if i % 1000 == 0:
            print("loss of step %i: %f" % (i, loss.detach().cpu().numpy()))

    X_eval, Y_eval = xor_data_generate(int(1e3))
    X_eval = X_eval.cuda()
    Y_eval = Y_eval.cuda()

    class_list = []
    x1_list = []
    x2_list = []
    colors = ['red', 'green']
    for i in range(int(1e3)):
        t = C(E(X_eval[i]))
        print("t:",t)
        if t[0][0] > t[0][1]:
            predict_label = 0
            class_list.append(0)

        else:
            predict_label = 1
            class_list.append(1)
        print("prediction:", predict_label)
        print("real label:", Y_eval[i])
        x1 = float(X_eval[i][0].cpu())
        x2 = float(X_eval[i][1].cpu())
        #print("x1:", x1)
        #print("x2:", x2)
        x1_list.append(x1)
        x2_list.append(x2)

    #fig = plt.figure(figsize=(8, 8))
    plt.scatter(x1_list, x2_list, c=class_list, cmap=matplotlib.colors.ListedColormap(colors))
    plt.savefig("test_c.png")


def Q_eval_vis():
    torch.manual_seed(233)
    torch.cuda.set_device(0)
    args = get_args()
    config = Config(
        state_dim=args.hidden,
        input_dim=args.input_dim,
        hidden=args.hidden,
        output_dim=args.num_classes,
        epsilon=args.epsilon
    )
    checkpoint = torch.load("cog396test_main_episode_280.tr")
    E = models.Shared_Encoder(config)
    E.load_state_dict(checkpoint['E_state_dict'])

    X_eval, Y_eval = xor_data_generate(int(1e3))
    X_eval = X_eval.cuda()
    Y_eval = Y_eval.cuda()

    Q = models.Simple_Q_Net(config)
    Q.load_state_dict(checkpoint['Q_state_dict'])

    E.cuda()
    Q.cuda()

    x1_list = []
    x2_list = []
    affs = []
    for i in range(1000):
        x_i = X_eval[i]
        s_i = E(x_i)
        q0, q1 = Q(s_i)     # q0: torch.
        affs.append(q1-q0)

        x1 = float(X_eval[i][0].cpu())
        x2 = float(X_eval[i][1].cpu())
        x1_list.append(x1)
        x2_list.append(x2)

    plt.scatter(x1_list, x2_list, c=affs, cmap='Blues')
    plt.savefig("policy_eval_280.png")


def main1():
    x = torch.tensor([[1,2,3], [4,5,6], [7,8,9]]).float()
    net = Simple_Pair_Net(x.shape[-1])

    y = net(x)

    print("y:", y)


def main2():
    torch.manual_seed(233)
    torch.cuda.set_device(0)
    args = get_args()
    config = Config(
        state_dim=args.hidden,
        input_dim=args.input_dim,
        hidden=args.hidden,
        output_dim=args.num_classes,
        epsilon=args.epsilon
    )
    checkpoint = torch.load("cog396test_main_episode_280.tr")
    C = models.SimpleNNClassifier(config)
    E = models.Shared_Encoder(config)
    C.load_state_dict(checkpoint['C_state_dict'])
    E.load_state_dict(checkpoint['E_state_dict'])
    C.cuda()
    E.cuda()

    X_eval, Y_eval = xor_data_generate(int(1e3))
    X_eval = X_eval.cuda()
    Y_eval = Y_eval.cuda()

    class_list = []
    x1_list = []
    x2_list = []
    colors = ['red', 'green']
    for i in range(int(1e3)):
        t = C(E(X_eval[i]))
        print("t:", t)
        if t[0][0] > t[0][1]:
            predict_label = 0
            class_list.append(0)

        else:
            predict_label = 1
            class_list.append(1)
        print("prediction:", predict_label)
        print("real label:", Y_eval[i])
        x1 = float(X_eval[i][0].cpu())
        x2 = float(X_eval[i][1].cpu())
        # print("x1:", x1)
        # print("x2:", x2)
        x1_list.append(x1)
        x2_list.append(x2)

    # fig = plt.figure(figsize=(8, 8))
    plt.scatter(x1_list, x2_list, c=class_list, cmap=matplotlib.colors.ListedColormap(colors))
    plt.savefig("train_c_280.png")


def main3():
    X_eval, Y_eval = xor_data_generate(1000)
    afn_list = []
    x1_list = []
    x2_list = []
    for i in range(1000):
        x1 = float(X_eval[i][0])
        x2 = float(X_eval[i][1])
        y = Y_eval[i]
        afn = np.random.uniform(0, 0.1)
        if y == 0 and (x2 < x1 + 0.3):
            d = float(abs((x2 - x1 + 0.3) / math.sqrt(2)))
            afn += np.random.normal(1-d, 1)
        x1_list.append(x1)
        x2_list.append(x2)
        afn_list.append(afn)

    plt.scatter(x1_list, x2_list, c=afn_list, cmap='Blues')
    plt.savefig("affn.png")


def main4():
    perplexities = [2030, 960, 884, 706, 713, 666, 653, 607, 608, 678, 702, 701]
    topics = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]

    plt.plot(topics, perplexities, label="LDA")

    tp = np.arange(0, 240, 20)
    y = np.array([608 for i in range(len(tp))])
    plt.plot(tp, y, 'r--', label="HDP")

    plt.title("Perplexity Scores for LDA and HDP Models")
    # plt.xlabel('number of data processed * %d'%avg_k, fontsize=14)
    plt.xlabel('number of topics', fontsize=14)
    plt.ylabel('perplexity score', fontsize=14)
    plt.legend()

    plt.savefig("lda_hdp.png", dpi=150)


def main5():
    X, y = twospirals(1000)


if __name__ == '__main__':
    #simple_train_C()
    #main2()
    #Q_eval_vis()
    main4()

