import pandas as pd
import numpy as np
np.random.seed(0)
# np.random.seed(7)
from numba import njit
import timeit
from math import sqrt

class HiddenLayer:
    # initialize random weights for layer, add one set of 
    # weights for the embedded bias 
    def __init__(self, n_inputs, tot_clas_len, momentum):
        self.weights = (  -2 * (np.random.random_sample((n_inputs+1, tot_clas_len))) + 1 )
        # apply normalization to increase convergence rate
        self.weights *= sqrt( 1 / (n_inputs+1) )
        self.momentum = momentum
        self.prev_delta = 0

    # with the previous layers output, calculate current layer output
    def forward(self, prev_output):
        # add a 1 to the end of the prev output vectors to simulate bias term
        bias_vec = np.full(prev_output.shape[0], 1)
        self.prev_output = np.hstack((prev_output, np.atleast_2d(bias_vec).T))
        # compute the actual input of the layer, z via the dot product 
        # from the previous layers activation output and the current layers
        # correlated weight
        self.actual_input_z = np.dot(self.prev_output, self.weights)
        # apply sigmoid activation to the outputs
        self.actv_out = (1/(1+np.exp(-self.actual_input_z)))

    # calculate gradients and the list of the activation functions multiplied
    # by weights so the previous layer can work directly with summed values 
    # rather than the function 
    def backward(self, passed):
        self.gradients = np.empty( (self.prev_output.shape[0],self.weights.shape[0],self.weights.shape[1]) )
        # the list that is corelated with the previous layers non-bias neurons
        intermediate = np.empty(( self.prev_output.shape[0] ,  self.weights.shape[1]))
        for example in range(passed.shape[0]):
            # fill matrix coloumn wise
            for j in range(self.gradients.shape[2]):
                # calulate derivative of activation function, sigmoid
                der_actv = self.actv_out[example][j] * (1 - self.actv_out[example][j])
                # multiplied by the sum of the derivative from previous layer
                sclar = passed[example][j] * der_actv 
                # fill one row of colomn at a time
                self.gradients[example][:,j].fill( sclar )
            # the derivatives of the current nodes are any line of the matrix
            intermediate[example] = np.copy( self.gradients[example][0] )
            # move row wise and multiply the previous output by current gradient
            for p in range(self.gradients.shape[1]):
                self.gradients[example][p] *= self.prev_output[example][p]
        # the list that is corelated with the previous layers non-bias neurons
        to_pass = np.zeros((self.prev_output.shape[0], self.weights.shape[0]-1))
        for j in range(self.prev_output.shape[0]):
            holding = np.zeros((self.weights.shape[0]-1, self.weights.shape[1]))
            for i in range(self.weights.shape[1]):
                # multiply first weight of every neuron by the current derivative
                holding[:,i] = intermediate[j][i] * self.weights[:-1][:,i]
            # sum g' * weights for every node to get sum of that node
            to_pass[j] = np.sum(holding, axis=1)
        # pass to the layer backwards
        return to_pass

    # sum the gradients from the batch then subtract from the weight
    def apply_grad(self):
        self.gradients = np.sum(self.gradients, axis=0)
        self.weights -= self.gradients
        self.weights -= (self.prev_delta * self.momentum)


class OutputLayerClassification:
    # initialize random weights for layer, add one set of 
    # weights for the embedded bias 
    def __init__(self, n_inputs, tot_clas_len, momentum):
        self.weights = (  -2 * (np.random.random_sample((n_inputs+1, tot_clas_len))) + 1 )
        # normalize and apply normalization
        self.weights *= sqrt( 1 / (n_inputs+1) )
        self.momentum = momentum
        self.prev_delta = 0


    # this function will calculate the softmax output
    def final_prediction(self, prev_output):
        # add a 1 to the end of the prev output vectors to simulate bias term
        bias_vec = np.full(prev_output.shape[0], 1)
        self.prev_output = np.hstack((prev_output, np.atleast_2d(bias_vec).T))
        # compute the actual input of the layer, z via the dot product 
        # from the previous layers activation output and the current layers
        # correlated weight
        self.actual_input_z = np.dot(self.prev_output, self.weights)
        # empty vector to hold the softmaxed values
        self.final_prob = np.empty(self.actual_input_z.shape)
        # for each input example in the mini-batch, compute softmax
        for i in range(len(self.actual_input_z)):
            # softmax
            e_x = np.exp(self.actual_input_z[i] - np.max(self.actual_input_z[i])) 
            self.final_prob[i] =  e_x / e_x.sum(axis=0)

    # this function will calculate the cross entropy loss
    # expected outputs is a one hot vector of the correct class value
    def calc_loss(self, expected_outputs):
        # calculate cross entropy loss
        one_h = (self.final_prob * expected_outputs)
        the_max = np.amax( one_h , axis=1)
        self.ce_loss = -1 * np.log(the_max)
        # calculate the mean of that loss
        self.mean_loss = np.mean(self.ce_loss)
        # calculate derivative of cross entropy
        self.loss_derv = self.final_prob - expected_outputs
        # find if the classification was successfull by comparing
        # highest probability to target 
        # max_indexes = np.argmax(self.final_prob, axis=1) - np.argmax(expected_outputs, axis=1)
        final_prob_arg_max = np.argmax(self.final_prob, axis=1)
        ex_output_arg_max  = np.argmax(expected_outputs, axis=1)
        total_wrong = 0
        for i in range(self.final_prob.shape[0]):
            if final_prob_arg_max[i] != ex_output_arg_max[i]:
                total_wrong+=1
                pass
        # wrong_array = np.absolute(max_indexes)
        self.correct = self.final_prob.shape[0] - total_wrong
        self.guess =  self.final_prob.shape[0]

    # calculate gradients and the list of the activation functions multiplied
    # by weights so the previous layer can work directly with summed values 
    # rather than the function 
    def backward(self, learning_rate):
        # gradients should be a 3d array with shape[0] being the amount of 
        # examples in a mini batch
        self.gradients = np.empty( (self.prev_output.shape[0],self.weights.shape[0],self.weights.shape[1]) )
        for example in range(self.actual_input_z.shape[0]):
            # this will hold y_hat - y * learning rate so that we can transpose
            # then push into gradient list
            hold = np.empty((self.weights.shape[1],self.weights.shape[0]))
            for i in range(self.gradients.shape[2]):
                # multiply by learning rate for smoother performance
                hold[i] =  ( self.prev_output[example] * self.loss_derv[example][i]  ) * learning_rate
            # hold is currently in the opposite orientation
            self.gradients[example] = hold.T
        # list to store the values that the previous layer will use
        to_pass = np.zeros((self.loss_derv.shape[0], self.weights.shape[0]-1))
        for j in range(self.loss_derv.shape[0]):
            holding = np.zeros((self.weights.shape[0]-1, self.weights.shape[1]))
            # generate the sum of g' times the weights for each node in the 
            # previous layer
            for i in range(self.loss_derv.shape[1]):
                abreviated_weight = self.weights[:-1][:,i]
                holding[:,i] = self.loss_derv[j][i] * abreviated_weight
            to_pass[j] = np.sum(holding, axis=1)
        return to_pass

    # sum the gradients from the batch then subtract from the weight
    def apply_grad(self):
        self.gradients = np.sum(self.gradients, axis=0)
        self.weights -= self.gradients
        self.weights -= (self.prev_delta * self.momentum)

class OutputLayerRegression:
    # initialize random weights for layer, add one set of 
    # weights for the embedded bias 
    def __init__(self, n_inputs, tot_clas_len, momentum):
        self.weights = (  -2 * (np.random.random_sample((n_inputs+1, tot_clas_len))) + 1 )
        # normalize and apply normalization
        # self.weights *= sqrt( 1 / (n_inputs+1) )
        self.momentum = momentum
        self.prev_delta = 0

    # this function will calculate the final target regression
    def final_prediction(self, prev_output):
        # add a 1 to the end of the prev output vectors to simulate bias term
        bias_vec = np.full(prev_output.shape[0], 1)
        self.prev_output = np.hstack((prev_output, np.atleast_2d(bias_vec).T))
        # compute the actual input of the layer, z via the dot product 
        # from the previous layers activation output and the current layers
        # correlated weight
        self.actual_input_z = np.dot(self.prev_output, self.weights)
        self.final_prob = self.actual_input_z

    # this function will calculate the Mean Absolute Error
    # expected outputs is [ regressionTarget ]
    def calc_loss(self, expected_outputs):
        # calculate mean absolute error
        self.final_loss = np.mean((abs(self.final_prob - np.reshape(expected_outputs,(-1,1)))))
        # calculate derivative of mean absolute error
        self.loss_derv = self.final_prob - np.reshape(expected_outputs,(-1,1))
    
    def backward(self, learning_rate):
        # gradients should be a 3d array with shape[0] being the amount of 
        # examples in a mini batch
        self.gradients = np.empty( (self.prev_output.shape[0],self.weights.shape[0],self.weights.shape[1]) )
        for example in range(self.actual_input_z.shape[0]):
            # this will hold y_hat - y * learning rate so that we can transpose
            # then push into gradient list
            hold = np.empty((self.weights.shape[1],self.weights.shape[0]))
            for i in range(self.gradients.shape[2]):
                # multiply by learning rate for smoother performance
                hold[i] =  ( self.prev_output[example] * self.loss_derv[example]  ) * learning_rate
            # hold is currently in the opposite orientation
            self.gradients[example] = hold.T
        # list to store the values that the previous layer will use
        to_pass = np.zeros((self.loss_derv.shape[0], self.weights.shape[0]-1))
        for j in range(self.loss_derv.shape[0]):
            holding = np.zeros((self.weights.shape[0]-1, self.weights.shape[1]))
            # generate the sum of g' times the weights for each node in the 
            # previous layer
            for i in range(self.loss_derv.shape[1]):
                abreviated_weight = self.weights[:-1][:,i]
                holding[:,i] = self.loss_derv[j][i] * abreviated_weight
            to_pass[j] = np.sum(holding, axis=1)
        return to_pass
    
    # sum the gradients from the batch then subtract from the weight
    def apply_grad(self):
        self.gradients = np.sum(self.gradients, axis=0)
        self.weights = self.weights - self.gradients
        pass