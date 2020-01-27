#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import array, copy, random, time
# import logging
import numpy as np
import pandas as pd
import seaborn
seaborn.set(style='whitegrid')

# imports for the BNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from deap import algorithms, base, creator, tools

IND_SIZE = 3
N_CYCLES = 1
BOUND_LOW, BOUND_UP = [217, 26, 1], [278, 44, 3]

creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin, n=IND_SIZE)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_temperature", random.randint, 217, 278)
toolbox.register("attr_speed", random.randint, 26, 44)
toolbox.register("attr_layer", random.randint, 1, 3)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_temperature,toolbox.attr_speed,toolbox.attr_layer), n=N_CYCLES)


# Structure initializers
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# load min and max values of the data to denormalize prediction data
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

def evaluate(vars):

    # for ii, item in enumerate(vars):
    #     vars[ii] = denormalize_max_min(vars[ii], max_x[ii], min_x[ii])

    # load BL model BNN
    BL_model = torch.load('BNN_BLmodel.pt')

    max_part_height = 4.2   # maximum part height mm

    # print(vars[2])

    # number of total layers = (maximum part height)/(height of a layer), i.e., 4.2 / (layer height)
    if vars[2] == 1:
        height = 0.42
    elif vars[2] == 2:
        height = 0.6
    elif vars[2] == 3:
        height = 0.7

    # print(vars)
    num_layers = np.int(max_part_height / height); # number of layers

    num_interfaces = 14     # number of interfaces per layer
    width = 0.8             # filament width in mm

    inp = [] # input to BNN to make predictions
    ycoord = 0.5 * height  # 0.5*height of a layer in mm
    iki_y = ycoord * 2

    # store inputs for GP(model disrepancy at each interface)
    for jj in range(1, num_layers + 1):
        for ii in range(1, num_interfaces + 1):
            # use x & y coordinates of vertical bonds as training data for the GP
            # Inp =[ Temperature, speed, height, x, y ]
            inp.append([vars[0], vars[1], height, ii * width, ycoord + (jj - 1) * iki_y])

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

    samples, noises = np.array(samples),  np.array(noises)
    means = (samples.mean(axis=0)).reshape(-1)

    aleatoric = (noises ** 2).mean(axis=0) ** 0.5
    epistemic = (samples.var(axis=0) ** 0.5).reshape(-1)
    total_unc = (aleatoric ** 2 + epistemic ** 2) ** 0.5

    print(means.mean(), total_unc.mean(), vars)
    # if means.mean()>0.7:
    #     print(means.mean(),total_unc.mean(),vars)

    # Dimensionless BL: non-dimensionalize the BL by dividing with the layer height
    dimensionless_mean_bl = means.mean()/height
    dimensionless_total_unc_bl = total_unc.mean()/height**2

    return dimensionless_mean_bl, dimensionless_total_unc_bl


def checkBounds(min, max):
    def decorator(func):
        def wrappper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max[i]:
                        print(child[i])
                        child[i] = max[i]
                    elif child[i] < min[i]:
                        print(child[i])
                        child[i] = min[i]
            return offspring
        return wrappper
    return decorator

toolbox.register("evaluate", evaluate)

# toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=10.0)
# toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=10.0, indpb=1.0/NDIM)


toolbox.register("mate", tools.cxUniform, indpb=0.50)
toolbox.register("mutate", tools.mutUniformInt, low=BOUND_LOW, up=BOUND_UP, indpb=0.50)


# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# toolbox.register("mate", tools.cxBlend, alpha=1.5)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=3, indpb=0.3)
toolbox.register("select", tools.selNSGA2)
# ref_points = tools.uniform_reference_points(nobj=2, p=12)
# toolbox.register("select", tools.selNSGA3WithMemory(ref_points))

# Bounds on the design variables
toolbox.decorate("mate", checkBounds([217, 26, 1], [278, 44, 3]))
toolbox.decorate("mutate", checkBounds([217, 26, 1], [278, 44, 3]))


toolbox.max_gen = 2  # num of generations
toolbox.pop_size = 50   # population size
toolbox.mut_prob = 0.2  # mutation probability

def main():
    random.seed(64)

    MU, LAMBDA = toolbox.pop_size, toolbox.pop_size
    pop = toolbox.population(n=MU)
    hof = tools.ParetoFront()
    # stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", np.mean, axis=0)
    # stats.register("std", np.std, axis=0)
    # stats.register("min", np.min, axis=0)
    # stats.register("max", np.max, axis=0)

    # create a stats to store the individuals not only their objective function values
    stats = tools.Statistics()
    stats.register("pop", copy.deepcopy)

    # Storing all the required information in the toolbox and using DEAP's
    # algorithms.eaMuPlusLambda function allows us to create a very compact -
    # albeit not a 100% exact copy of the original- implementation of NSGA-II.
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                              cxpb=1-toolbox.mut_prob, mutpb=toolbox.mut_prob,
                                             ngen=toolbox.max_gen,
                              stats=stats, halloffame=hof, verbose=True)

    return pop, stats, hof, logbook

# # Plot the Pareto front
# if __name__ == "__main__":
#     pop, stats, hof, logbook = main()
#
#     import matplotlib.pyplot as plt
#     import numpy
#
#     front = numpy.array([ind.fitness.values for ind in pop])
#     plt.scatter(front[:,0], front[:,1], c="b")
#     plt.axis("tight")
#     plt.show()


def animate(frame_index, logbook):
    'Updates all plots to match frame _i_ of the animation.'
    ax.clear()
    plot_colors = seaborn.color_palette("Set1", n_colors=10)
    fronts = tools.emo.sortLogNondominated(logbook.select('pop')[frame_index],
                                           len(logbook.select('pop')[frame_index]))
    for i, inds in enumerate(fronts):
        par = [toolbox.evaluate(ind) for ind in inds]
        df = pd.DataFrame(par)
        df.plot(ax=ax, kind='scatter', label='Front ' + str(i + 1),
                x=df.columns[0], y=df.columns[1], alpha=0.47,
                color=plot_colors[i % len(plot_colors)])

    ax.set_title('$t=$' + str(frame_index))
    ax.set_xlabel('$f_1(\mathbf{x})$');
    ax.set_ylabel('$f_2(\mathbf{x})$')
    return []

# Animation
if __name__ == "__main__":
    pop, stats, hof, logbook = main()

    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = '\\usepackage{libertine}\n\\usepackage[utf8]{inputenc}'

    from matplotlib import animation
    from IPython.display import HTML

    from IPython.display import set_matplotlib_formats
    set_matplotlib_formats('retina')
    
    # get the Pareto fronts in the population (pop).
    fronts = tools.emo.sortLogNondominated(pop, len(pop))

    fig = plt.figure(figsize=(4,4))
    ax = fig.gca()
    anim = animation.FuncAnimation(fig, lambda i: animate(i, logbook),
                                   frames=len(logbook), interval=60,
                                   blit=True)
    plt.close()
    HTML(anim.to_html5_video())

