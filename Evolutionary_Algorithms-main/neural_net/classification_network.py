import pandas as pd
import numpy as np
np.random.seed(0)
from math import sqrt
from layers import HiddenLayer, OutputLayerClassification
from partition_data import PartitionData

# this class is an object of the entire network and contains functions to
# coordinate both the training and testing of the network
class ClassificationNetwork():
    # save the inputs, calculate number of inputs, calc the 
    # number of unique classes, initialize the layer list
    def __init__(self, inputs, momentum):
        self.inputs = inputs
        self.num_inputs = inputs.shape[1] -1
        self.num_unique_class = len(inputs['target_class'].iloc[0])
        self.momentum = momentum
        self.layer_list = []
    
    # this func receives a dataframe that contains all the input data 
    # and a one hot encoded vector of the class
    def mini_batches(self, dataframe, batch_size):

        if batch_size == "max":
            batch_size = dataframe.shape[0]
        # shuffle data  
        dataframe = dataframe.sample(frac=1)
        # create group of data frame based on batch size
        hmm = dataframe.groupby(np.arange(len(dataframe))//   batch_size    )
        target_class_per_batch=[[] for x in range(len(hmm))]
        batch_list=[]
        # for each dataframe group, separate data vs labels
        for ind, group in enumerate(hmm):
            for each in group[1]['target_class'].iteritems():
                target_class_per_batch[ind].append(np.array(each[1]))
            target_class_per_batch[ind] = np.array(target_class_per_batch[ind])
            batch_list.append( group[1].drop('target_class', axis=1).to_numpy() )
        # return a list of mini batched and the coorelated target classes
        return batch_list, target_class_per_batch

    # add hidden layer, specify num of neurons
    def add_Hidden(self, num_neurons):
        # if there are no previous hidden layers, the number of inputs is the number
        # of features from the input layer, else it is the amount of neurons 
        # from the previous layer
        if not self.layer_list:
            num_inputs = self.num_inputs
        else:
            num_inputs = self.prevois_num_nueron
        # initialize layer 
        self.layer_list.append( HiddenLayer(num_inputs, num_neurons, self.momentum) )
        # record amount of neurons so next layer can use as input
        self.prevois_num_nueron = num_neurons
        pass
    # add output layer. no paramaters
    def add_Out(self):
        # if there are no hidden layers, the number of inputs is the number
        # of features from the input layer, else it is the amount of neurons 
        # from the previous layer
        if not self.layer_list:
            num_inputs = self.num_inputs
        else:
            num_inputs = self.prevois_num_nueron
        # initialize layer
        self.out_layer = OutputLayerClassification(num_inputs, self.num_unique_class, self.momentum) 
        pass

    # forward pass for the entire network
    def forward(self, X, target_class_per_batch, mni_btch_i=0):
        for lay_num in range(len(self.layer_list)):
            # if first hidden layer, send raw input in
            if lay_num == 0:
                self.layer_list[lay_num].forward( X[mni_btch_i] )
                # save the activated output as the new input for next layer
                X[mni_btch_i] = self.layer_list[lay_num].actv_out
            else:
                # send previous layers activated output and reassign the next input
                self.layer_list[lay_num].forward( self.layer_list[lay_num - 1].actv_out )
                X[mni_btch_i] = self.layer_list[lay_num].actv_out


        # the forward pass and compute the softmax
        self.out_layer.final_prediction(X[mni_btch_i])
        # calculate the loss using cross entropy
        self.out_layer.calc_loss(target_class_per_batch[mni_btch_i])
        
    def backprop(self):
        # get derivative of activation function for layer -> to_pass
        to_pass = self.out_layer.backward(learning_rate=0.1)
        # for every hidden layer, compute derivative based on
        # previous derivative, pass values back 
        for lay_i in range( len(self.layer_list)-1, -1, -1):
            to_pass = self.layer_list[ lay_i ].backward( to_pass )
            pass
        
        self.out_layer.apply_grad()
        # apply gradient for every hidden layer
        for lay_i in range(len(self.layer_list)):
            self.layer_list[ lay_i ].apply_grad()

    # run forward then back prop for every mini batch for every epoch
    def train(self, num_epochs, batch_divisor ):
        for _ in range(num_epochs):
            X, target_class_per_batch = self.mini_batches(self.inputs, batch_size= batch_divisor)
            # for each mini batch
            # KEEP IN MIND X[mni_btch_i] will hold the activated output of the previous layer
            # if there are no hidden layers, it holds the example outputs
            tot_mean_loss = 0
            total_correct = 0
            total_guess = 0
            for mni_btch_i in range(len(X)):
                # forward on current data
                self.forward(X, target_class_per_batch, mni_btch_i)
                self.backprop()
                # get totals of all metrics
                total_correct += self.out_layer.correct
                total_guess += self.out_layer.guess
                tot_mean_loss += self.out_layer.mean_loss
            # print and write to file
            # print( "total mean cross entropy loss: ", tot_mean_loss / len(X) )
            # print(f"Guessed {total_correct}/{total_guess} correctly, or {round((total_correct/total_guess)*100 ,2)}%")
            # with open("performace_data.txt", "a") as w:
            #     w.write(str(tot_mean_loss / len(X)))
            #     w.write(",")
            #     w.write(str(total_correct/total_guess))
            #     w.write("\n")

    # run the validation set through the network and compute loss 
    def test(self, inputs):
        # get single batch in correct format
        X, target_class_per_batch = self.mini_batches(inputs, batch_size= 'max')
        self.forward(X, target_class_per_batch)
        # print results
        # print( "total mean cross entropy loss: ", self.out_layer.mean_loss  )
        self.correct_perc = round((self.out_layer.correct/self.out_layer.guess)*100 ,2)
        # print(f"Guessed {self.out_layer.correct}/{self.out_layer.guess} correctly, or {self.correct_perc}%")
