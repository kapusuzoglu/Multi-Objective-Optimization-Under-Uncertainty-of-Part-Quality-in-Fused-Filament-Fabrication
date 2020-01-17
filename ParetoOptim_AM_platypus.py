from platypus import NSGAII, Problem, Real
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# load min and max values of training data to denormalize prediction data
with open('maxmin.pickle', 'rb') as f:
    [max_x, min_x, max_y, min_y] = pickle.load(f)

def normalize_max_min(data, data_max, data_min):
    return (data-data_min) / (data_max-data_min)

def denormalize_max_min(data, data_max, data_min):
    return data * (data_max-data_min) + data_min

class MC_Dropout_Model(nn.Module):
    def __init__(self, input_dim, output_dim, num_units, drop_prob):
        super(MC_Dropout_Model, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drop_prob = drop_prob

        # network with two hidden and one output layer
        self.layer1 = nn.Linear(input_dim, num_units)
        self.layer2 = nn.Linear(num_units, num_units)
        self.layer3 = nn.Linear(num_units, 2 * output_dim)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(-1, self.input_dim)

        x = self.layer1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.drop_prob, training=True)

        x = self.layer2(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.drop_prob, training=True)

        x = self.layer3(x)

        return x

def BNN_BL_obj(vars):

    # load BL model BNN
    BL_model = torch.load('BNN_BLmodel.pt')

    max_part_height = 4.2   # maximum part height mm

    # number of total layers is maximum part height / height of each layer, i.e., 4.2 / (layer height)
    if 0.42 < vars[2] < 0.51:
        vars[2] = 0.42
    elif 0.51 < vars[2] < 0.6:
        vars[2] = 0.6
    elif 0.6 < vars[2] < 0.65:
        vars[2] = 0.6
    elif 0.65 < vars[2] < 0.7:
        vars[2] = 0.7

    # print(vars)
    num_layers = np.int(max_part_height / vars[2]);
    num_interfaces = 14  # number of interfaces per layer
    width = 0.8 # filament width in mm

    inp = []
    ycoord = 0.5 * vars[2]  # 0.5*height of a layer in mm
    iki_y = ycoord * 2

    # store inputs for GP(model disrepancy at each interface)
    for jj in range(1, num_layers + 1):
        for ii in range(1, num_interfaces + 1):
            # use x & y coordinates of vertical bonds as training data for the GP
            # Inp =[ Temperature, speed, height, x, y ]
            inp.append([vars[0], vars[1], vars[2], ii * width, ycoord + (jj - 1) * iki_y])

    # Convert built Python lists to a Numpy array.
    inp = np.array(inp, dtype='float32')

    # normalize data
    inp = normalize_max_min(inp, max_x, min_x)

    x_pred = torch.tensor(inp)  # convert to torch tensor

    samples = []
    noises = []
    for i in range(100):
        preds = BL_model.forward(x_pred).cpu().data.numpy()
        samples.append(denormalize_max_min(preds[:, 0], max_y, min_y))
        noises.append(denormalize_max_min(np.exp(preds[:, 1]), max_y, min_y))

    samples = np.array(samples)
    noises = np.array(noises)
    means = (samples.mean(axis=0)).reshape(-1)

    aleatoric = (noises ** 2).mean(axis=0) ** 0.5
    epistemic = (samples.var(axis=0) ** 0.5).reshape(-1)
    total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5

    # print(means.mean(),total_unc.mean())
    return [-means.mean(), total_unc.mean()]

class AM(Problem):
    def __init__(self):
        # we call Problem(2, 2, 2) to create a problem with
        # two decision variables, two objectives, and two constraints, respectively
        super(AM, self).__init__(3, 2)
        self.types[:] = [Real(210, 280), Real(15, 50), Real(0.42, 0.7)]
        # self.constraints[:] = "<=0"

    def evaluate(self, solution):
        t = solution.variables[0]
        v = solution.variables[1]
        h = solution.variables[2]
        solution.objectives[:] = BNN_BL_obj(np.array([t,v,h]))
        # solution.constraints[:] = [-x + y - 1, x + y - 7]


algorithm = NSGAII(AM())
algorithm.run(1000)


# plot the results using matplotlib
import matplotlib.pyplot as plt

plt.scatter([-s.objectives[0] for s in algorithm.result],
            [s.objectives[1] for s in algorithm.result])
# plt.xlim([0, 1.1])
# plt.ylim([0, 1.1])
plt.xlabel("$f_1(x)$")
plt.ylabel("$f_2(x)$")
plt.show()