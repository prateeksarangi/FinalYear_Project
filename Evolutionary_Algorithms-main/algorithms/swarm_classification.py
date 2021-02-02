# John Rigoni
# This file is the implimentation of particle swarm optimization for classification
# It essentially wraps the previous neural network that was implimented for 
# project 3
# The class Particle contains a state which are neural net weights in this case,
# a velocity, and current best among other fields. It has a function to evaluate
# performance and also to update to the next time step.
# The class Swarm hold the population of Particles. It contains the logic that 
# allows for the swarm to update. It also contains a function to allow for the 
# testing of the test dataset.
# The class Network simply allows the instantiation of a neural net with an arbitrary
# number of layers and nodes.
# 
# The __main__ function instantiates a Swarm object, with the training set and parameters.
# The Swarm object will then instantiate a Particle object which will then instantiate
# a network object.
# 
# The data sets are set up in a standardized dataframe object, identical to the previous project

from classification_network import ClassificationNetwork
from regression_network import RegressionNetwork
from partition_data import PartitionData
import data_sets
import pandas as pd
import numpy as np
from random import uniform, seed, normalvariate
from copy import deepcopy
from statistics import mean, stdev
seed(0)

# This class is the unit for operations to be performed on. It contains a state,
# a velocity, and a previous best state. This is the representation of the neural net
# that will be progressively updated. Contains an evaluate function to generate loss
# and an evolve function to update.
class Particle():
    # initialize the object.
    # Input:
    #   Xtrain : training set in standard dataframe format
    #        in order to initialize correct amount of weights
    #   C1 & C2 : parameter for cognitive and social componenets
    #   omega : parameter for inertia component
    def __init__(self, Xtrain, C1, C2, omega):
        # save dataset
        self.Xtrain = Xtrain
        network = Network(self.Xtrain)
        self.network = network.network
        self.state = []
        self.velocity = []
        self.best_state = []
        self.err_best = 1000
        self.cur_err = 1000
        self.C1 = C1
        self.C2 = C2
        self.omega = omega
        self.particle_layer_list = []
        # fill state and network list with the networks weights
        if self.network.layer_list:
            for i in range(len(self.network.layer_list)):
                self.state.append( self.network.layer_list[i].weights )
                self.particle_layer_list.append( self.network.layer_list[i] )

        # fill state with nn weights
        self.state.append( self.network.out_layer.weights )
        self.particle_layer_list.append ( self.network.out_layer )
        # copy state to ascertain correct array structure
        self.velocity = deepcopy(self.state)
        self.best_state = deepcopy(self.state)
        # fill velocity with random variables
        for a in range(len(self.state)):
            for b in range(a):
                for c in range(b):
                    self.velocity[a][b][c] = normalvariate(mu=0, sigma=1)

    # evaluate the current state of the network on the training set,
    # save if personal best
    def evaluate(self):
        # test net
        self.network.test(self.Xtrain)
        self.cur_err = self.network.out_layer.mean_loss
        # save best
        if self.cur_err < self.err_best:
            self.err_best = self.cur_err
            self.best_state = self.state

    # calculate new velocity, then add to position to get new state
    def update(self, best_part):
        # bounds for velocity
        max_vel = 1
        min_vel = -1
        # iterate feature wise
        for a, first in enumerate(self.velocity):
            for b, second in enumerate(first):
                for c, third in enumerate(second):
                    # calculate new velocity
                    inertia = self.omega * self.velocity[a][b][c]
                    cognitive = self.C1 * uniform(0,1) * (self.best_state[a][b][c] - self.state[a][b][c])
                    social = self.C2 * uniform(0,1) * (best_part.state[a][b][c] - self.state[a][b][c])
                    new_vel = inertia + cognitive + social
                    # clamp velocity if necessary
                    if new_vel > max_vel:
                        new_vel = max_vel
                    if new_vel < min_vel:
                        new_vel = min_vel
                    # update each velocity value
                    self.velocity[a][b][c] = new_vel
        # update state
        self.state = [x + y for x, y in zip(self.state, self.velocity)]
        # update weights of nn
        for i in range(len(self.velocity)):
            self.particle_layer_list[i].weights = self.state[i]

# This class instantiates and stores all the Particle objects. It contains
# the logic to minimize the loss function and test on the test set
class Swarm():
    # initialize object, populating the population list with particles
    # Input:
    #   Xtrain : training set in standard dataframe format
    #        in order to initialize correct amount of weights
    #   C1 & C2 : parameter for cognitive and social componenets
    #   omega : parameter for inertia component
    def __init__(self, Xtrain, C1, C2, omega, particles):
        self.particle_count = particles
        self.particle_list = []
        # instantiate list
        for _ in range(self.particle_count):
            self.particle_list.append( Particle(Xtrain, C1, C2, omega) )

    # the logic necessary to calculate the global best particle, then update
    # each particle in the swarm based on that. 
    # Input:
    #   iterations: the amount of times to update
    def minimize(self, iterations):
        for _ in range(iterations):
            # initialize bunk values
            self.global_best_index = 1000
            self.global_best_total = 1000
            # go through every particle and record the best performing one
            for i in range(self.particle_count):
                self.particle_list[i].evaluate()
                if self.particle_list[i].network.out_layer.mean_loss < self.global_best_total:
                    self.global_best_index = i
                    self.global_best_total =  self.particle_list[i].network.out_layer.mean_loss
            
            # print("best min: ",self.global_best_total)
            # update each particle with the best particle
            for i in range(self.particle_count):
                self.particle_list[i].update(self.particle_list[self.global_best_index])
        
        # print best loss
        p = self.particle_list[self.global_best_index].network.out_layer.final_prob
        # print(p)
        print(f"Guessed {self.particle_list[self.global_best_index].network.out_layer.correct}/{self.particle_list[self.global_best_index].network.out_layer.guess} correctly, or {self.particle_list[self.global_best_index].network.correct_perc}%")

    # test on the test set and save loss
    def test(self, Xtest):
        print("Testing\n")
        # test
        self.particle_list[self.global_best_index].network.test(Xtest)
        p = self.particle_list[self.global_best_index].network.out_layer.mean_loss
        print(p)
        print(f"Guessed {self.particle_list[self.global_best_index].network.out_layer.correct}/{self.particle_list[self.global_best_index].network.out_layer.guess} correctly, or {self.particle_list[self.global_best_index].network.correct_perc}%")
        # save loss
        self.cor_perc = self.particle_list[self.global_best_index].network.correct_perc
        pass
        
# This class merely allows for the instantiation of a neural network
# with an arbitrary amount of layers and nodes. Provides useful functions
# primarily testing
# Input:
#   Xtrain: Train dataset in oder to create appropriate amount of layers and nodes
class Network():
    def __init__(self, Xtrain):

        self.network = ClassificationNetwork(inputs= Xtrain, momentum =0.5)
        self.network.add_Hidden(num_neurons=8)
        self.network.add_Hidden(num_neurons=5)
        self.network.add_Out()

# Main function. Generate 10 folds worth of data then
# evolve and test over those 10 folds. Record loss from
# those 10 folds and calculate mean and standard deviation.
# Print and write to appropriate file.
if __name__ == "__main__":
    # Import the correct data set

    # X = data_sets.breast_cancer_parse.get_data()
    X = data_sets.soybean_parse.get_data()
    # X = data_sets.glass_parse.get_data()
    
    # shuffle data set
    X = X.sample(frac=1)
    # generate 10 random folds of train and test sets
    part = PartitionData(X)

    total_cor_per = []
    for fold in range(10):
        Xtrain = part.train[fold]
        Xtest = part.test[fold]
        # instantiate class to perform particle swarm optimization
        swarm = Swarm(Xtrain, C1=0.4, C2=1, omega=0.9, particles=50 )
        # perform minimization for given amount of iterations
        swarm.minimize(iterations=55)
        # calculate the loss on the test set
        swarm.test(Xtest)
        # add loss to list
        total_cor_per.append( swarm.cor_perc )

    # calculate mean and stdev, write to file, print
    meaned = mean(total_cor_per)
    std_deved = stdev(total_cor_per)
    # with open("data_out/soybean/swarm_out.txt", "a") as w:
    #     w.write("2 Hidden Layers\n")
    #     w.write(f"{meaned} {std_deved}\n")
    print("\naverage correct")
    print("Mean: ", meaned)
    print("Std Dev: ", std_deved)
