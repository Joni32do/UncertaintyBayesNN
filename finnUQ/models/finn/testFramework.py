#This file is used for training and evaluating different models.
#Compared will be their performance, training/evaluation time NSSR/AOC and UQ

#test set D is without noise
#test set D_noise is with noise

listOfArchitectures = [[1, 32, 1],
                       [1, 8, 8, 1],
                       [1, 4, 9, 4, 1],
                       [1, 8, 4, 8, 1]]

pretraining = True
sort_bias = True



