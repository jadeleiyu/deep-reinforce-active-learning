import numpy as np
import matplotlib.pyplot as plt


def xor_data_generate(m):
    """
    generate m (x,y) data pairs by x-or distribution
    :param m:
    :return:
    """
    X = []
    Y = []
    for i in range(m):
        x1 = np.random.uniform(low=0, high=1)
        x2 = np.random.uniform(low=0, high=1)

        if (x1 > 0.5 and x2 > 0.5) or (x1 < 0.5 and x2 < 0.5):
            y = 1
        else:
            y = 0

        X.append([x1, x2])
        Y.append(y)

    #return torch.tensor(X).view(m,-1), torch.tensor(Y).view(m,-1)
    return X, Y


def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))








