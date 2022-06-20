#!/usr/bin/env python3

import array
import random

import numpy
from functools import reduce
from copy import copy
from functools import partial

from deap import algorithms
from deap import base
from deap import creator
from deap import tools



distances = [
    # A   B   C   D   E
    [ 0,  7,  9,  8, 20],
    [ 7,  0, 10,  4, 11],
    [ 9, 10,  0, 15,  5],
    [ 8,  4, 15,  0, 17],
    [20, 11,  5, 17,  0],
]

filename = "data.csv"
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 4)
toolbox.register("attr_shuffle", random.shuffle, [1, 2, 3, 4, 5])

arr = [0, 1, 2, 3, 4]

def generate_genome():
    global arr
    if len(arr) == 0:
        arr = [0, 1, 2, 3, 4]
        random.shuffle(arr)
    return arr.pop()

toolbox.register("attr_take1", generate_genome)

funcs = [
    toolbox.attr_take1,
]
# Structure initializers
x = partial(random.sample, range(5), 5)
toolbox.register("individual", tools.initCycle, creator.Individual, funcs, 5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def save_to_file(filename, line):
    with open(filename, "a") as log_file:
        log_file.write(line + "\n")

def evalTravelerTree(individual):
    travel_sum = sum([distances[individual[i]][individual[i+1]] for i in range(len(individual)-1)])
    save_to_file(filename, f"{individual},{travel_sum}")

    return travel_sum,

def mate(ind1, ind2):
    i1 = copy(ind1[:3])
    i1.append(ind1[4])
    i1.append(ind1[3])

    i2 = copy(ind2[:3])
    i2.append(ind2[4])
    i2.append(ind2[3])
    return ind1, ind2

toolbox.register("evaluate", evalTravelerTree)
toolbox.register("mate", mate)
# toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():

    random.seed(64)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=30,
                                   stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof

if __name__ == "__main__":
    main()
