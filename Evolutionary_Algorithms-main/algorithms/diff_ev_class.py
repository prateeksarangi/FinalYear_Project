# John Rigoni
# This file is the implimentation of differential evolution for classification
# It essentially wraps the previous neural network that was implimented for 
# project 3
# The class Organism is is essentially a unique neural net object that has some other useful
# variables. 
# The class Nature holds the population of Organisms, doing the selection, crossover, and mutation
# Eventually finding the best Organism, it has a function to evaluate usuing the test set.
# The class Network simply allows the instantiation of a neural net with an arbitrary
# number of layers and nodes.
# 
# The __main__ function instantiates a Nature object, with the training set,
# population size, and pr arguments. 
# The Nature object will then instantiate the appropriate amount of Organism
# objects which will instantiate one Network object each. 
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

# This class represents x in the set X. It will instantiate a Network
# Object and copy the weights into self.state to allow for easy operations.
# Also providing an evaluation function to ascertain the final loss of the network
class Organism():
    # initialize the object.
    # Input: 
    #   Xtrain = training set in standard dataframe format
    #       in order to initialize correct amount of weights
    def __init__(self, Xtrain):
        # save dataset
        self.Xtrain = Xtrain
        network = Network(self.Xtrain)
        self.network = network.network
        self.state = []
        self.organism_layer_list = []
        # fill state and network list with the networks weights
        if self.network.layer_list:
            for i in range(len(self.network.layer_list)):
                self.state.append( self.network.layer_list[i].weights )
                self.organism_layer_list.append( self.network.layer_list[i] )
        self.state.append( self.network.out_layer.weights )

    # evaluate the current state of the network on the training set
    def evaluate(self):
        # test and save result
        self.network.test(self.Xtrain)
        self.netw_loss = self.network.out_layer.mean_loss

# This class instantiates and stores all the organism objects in a list named population,
# performing the evolution process on them as many times as designated.
class Nature():
    # initialize object, populating the population list with organisms
    # Inputs:
    #   Xtrain: Train Dataset
    #   pop_size: Population Size
    #   pr: Pr parameter for crossover 
    def __init__(self, Xtrain, pop_size, pr):
        # save data set
        self.Xtrain = Xtrain
        # parameter population size
        self.pop_size = pop_size
        self.pr = pr
        self.pop_list = []
        # fill population list
        for _ in range(pop_size):
            self.pop_list.append( Organism(Xtrain) )

    # primary function that handels selection, crossover, mutation,
    # updating the population each generation/iteration
    # Inputs:
    #   Iterations: Number of Generations 
    #   beta: Parameter to Calculate Trial Vector 
    def evolve(self, iterations = 60, beta = 1):
        # loop for num of generations
        for _ in range(iterations):
            self.to_pop_list = []
            # go through each member of the population
            for i in range(self.pop_size):
                target = self.pop_list[i]
                distinct_list = []          
                # grab three random organisms
                index_list = []
                while (True):
                    if len(index_list) == 3:
                        break
                    rand_index = np.random.randint(0, len(self.pop_list))
                    # make sure its new
                    if rand_index not in index_list and rand_index != i:
                        index_list.append(rand_index)
                # add 3 organisms to list
                for j in range(3):
                    distinct_list.append( self.pop_list[ index_list[j] ] )
                # calculate the second term of the trial vector equation
                sec_term = []
                for first in range(len(self.pop_list[0].state)):
                    arr1 = self.pop_list[ index_list[1] ].state[first]
                    arr2 = self.pop_list[ index_list[2] ].state[first]
                    tot = np.subtract(arr1, arr2) * beta
                    sec_term.append( tot )
                # calculate full trail vector
                final_mut = []
                for first in range(len(self.pop_list[0].state)):
                    arr1 = self.pop_list[ index_list[0] ].state[first]
                    arr2 = sec_term[first]
                    tot = np.add(arr1, arr2)
                    final_mut.append( tot )
                # cross over the target and trail vector
                self.cross_over(target.state, final_mut)
                # create an organism object from x prime
                vs_target = Organism(Xtrain)
                vs_target.state = self.arr_hold
                # evaluate the organisms and save loss
                target.evaluate()
                target_loss = target.netw_loss
                vs_target.evaluate()
                vs_target_loss = vs_target.netw_loss

                # compare performace of target vs crossed over and
                # add the better to the next population
                if  target_loss < vs_target_loss:
                    self.to_pop_list.append( target )
                else:
                    self.to_pop_list.append( vs_target )
            # update population
            self.pop_list = deepcopy( self.to_pop_list )

    # test the best network on the test set
    # Input:
    #   Xtest: Test Dataset       
    def best_test(self, Xtest):
        # find the best network
        self.best_index = 1000
        best_error = 1000
        for i in range(self.pop_size):
            if self.pop_list[i].netw_loss < best_error:
                best_error = self.pop_list[i].netw_loss
                self.best_index = i
        # test set on best network
        self.pop_list[self.best_index].network.test(Xtest)
        self.best_loss = self.pop_list[self.best_index].network.out_layer.mean_loss
        print(f"Guessed {self.pop_list[self.best_index].network.out_layer.correct}/{self.pop_list[self.best_index].network.out_layer.guess} correctly, or {self.pop_list[self.best_index].network.correct_perc}%")
        self.cor_perc = self.pop_list[self.best_index].network.correct_perc

                    
    # Binomial Crossover. Save result in self.arr_hold
    # Input:
    #   arr1: Target Vector
    #   arr2: Trail Vector
    def cross_over(self, arr1, arr2):
        # copy in order to use the structure of the array
        self.arr_hold = deepcopy(arr1)
        for first in range(len(arr1)):
            for second in range(len(arr1[first])):
                for third in range(len(arr1[first][second])):
                    # generate random number and compare to pr
                    rand = np.random.uniform(0,1)
                    if rand < self.pr:
                        self.arr_hold[first][second][third] = arr1[first][second][third]
                    else:
                        self.arr_hold[first][second][third] = arr2[first][second][third]
        

# This class merely allows for the instantiation of a neural network
# with an arbitrary amount of layers and nodes. Provides useful functions,
# primarily testing
# Input:
#   Xtrain: Train dataset in order to create appropriate amount of input layers nodes
class Network():
    def __init__(self, Xtrain):

        self.network = ClassificationNetwork(inputs= Xtrain, momentum =0.5)
        self.network.add_Hidden(num_neurons=6)
        self.network.add_Hidden(num_neurons=4)
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
        # instantiate class to perform differential evolution
        nature = Nature(Xtrain, pop_size= 60, pr=0.5)
        # perform evolution for given amount of iterations
        nature.evolve(iterations=100)
        # calculate the loss on the test set
        nature.best_test(Xtest)
        # add loss to list
        total_cor_per.append( nature.cor_perc )
    # calculate mean and stdev, write to file, print
    meaned = mean(total_cor_per)
    std_deved = stdev(total_cor_per)
    # with open("data_out/soybean/diff_ev_out.txt", "a") as w:
    #     w.write("2 Hidden Layers\n")
    #     w.write(f"{meaned} {std_deved}\n")
    print("\naverage correct")
    print("Mean: ", meaned)
    print("Std Dev: ", std_deved)
