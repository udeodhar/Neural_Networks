###########################################################################################
#Title		: Implementation of Neural Networks using Python. 
#Author		: Unmesh Mahendra Deodhar
#Description: It is the implementation of Neural Networks. Trained MNIST data for detecting
#			  handwritten digits. Achieved accuracy around 95%.
###########################################################################################

import numpy as np
import matplotlib.pyplot as plot
import pandas as ps
import math as mth

def main():
	#Hyperparameters...
    leareningRate = 0.1
    epochs = 50
    noofHiddenUnits = 100
    momentum = 0.9
    examples = 60000

    print "Reading data"
    training_examples = np.array(ps.read_csv("mnist_train.csv", header=None), np.float)
    testing_examples = np.array(ps.read_csv("mnist_test.csv",header=None), np.float)

    training_bias = np.ones((examples,1), dtype=float);
    testing_bias = np.ones((10000, 1), dtype=float);

    training_examples[:, 1:] = (training_examples[:, 1:]/255.0)
    testing_examples[:, 1:] = (testing_examples[:, 1:] / 255.0)

    training_examples = np.append(training_examples, training_bias, axis=1)
    testing_examples = np.append(testing_examples, testing_bias, axis=1)
    print "data read and cleaned"
	
    outputLayer = np.zeros(noofHiddenUnits+1)

    outputLayer[0] = 1

    weightsI2H = np.random.uniform(-0.05, 0.05, (785, noofHiddenUnits))
    weightsH2O = np.random.uniform(-0.05, 0.05, (noofHiddenUnits+1, 10))

    oldDeltaH = np.zeros((noofHiddenUnits, 785))
    oldDeltaK = np.zeros((noofHiddenUnits+1, 10))

    expected_output_vector = np.zeros((examples, 10),float) + 0.1

    training_accuracy = np.zeros(epochs,float)
    testing_accuracy = np.zeros(epochs, float)


    for i in range(examples):
        expected_output_vector[i][int(training_examples[i][0])] = 0.9

    hidden_layer_activations =np.zeros(noofHiddenUnits + 1)
    hidden_layer_activations[0] = 1
    for epoch in range(epochs):
        # Reset confusion matrices
        training_confusion_matrix = np.zeros((10, 10), int )
        testing_confusion_matrix = np.zeros((10, 10), int )
    
        print "Epoch: ",epoch
        for i in range(examples):

     		# Feed forward
            hidden_layer_activations[1:] = (1 / (1 + np.exp(-1*np.dot(training_examples[i][1:], weightsI2H))))

            outputLayer[1:] = hidden_layer_activations[1:]
            output_layer_activations = (1 / (1 + np.exp(-1*np.dot(outputLayer,  weightsH2O))))
            output_layer_error_terms = (output_layer_activations *
                                            (1 - output_layer_activations) * (expected_output_vector[i] -
                                                                              output_layer_activations))
            hidden_layer_error_terms = (hidden_layer_activations[1:] * (1 - hidden_layer_activations[1:])* np.dot(weightsH2O[1:, :], output_layer_error_terms))


            deltaK = leareningRate * (np.outer( hidden_layer_activations, output_layer_error_terms)) + (momentum * oldDeltaK)
            deltaH = leareningRate * np.outer(hidden_layer_error_terms, training_examples[i][1:]) + (
                                        momentum * oldDeltaH)
            weightsH2O = weightsH2O + deltaK
            oldDeltaK = deltaK
            weightsI2H = weightsI2H + deltaH.T
            oldDeltaH = deltaH

            training_confusion_matrix[int(training_examples[i][0])][int(np.argmax(output_layer_activations))]+=1

        training_accuracy[epoch] = (float((sum(training_confusion_matrix.diagonal()))/60000.0)*100.0)
        print "Epoch ", epoch, ": ", "Training Accuracy: ", training_accuracy[epoch], "%"

		#Claculating for test data
        for i in range(10000):
            # Feed forward pass input to output layer through hidden layer
            hidden_layer_activations[1:] = (
            1 / (1 + np.exp(-1 * np.dot(testing_examples[i][1:], weightsI2H))))
            # Forward propagate the activations from hidden layer to output layer
            outputLayer[1:] = hidden_layer_activations[1:]
            # calculate dot product for output layer
            # apply sigmoid function to sum of weights times inputs
            output_layer_activations = (1 / (1 + np.exp(-1 * np.dot(outputLayer, weightsH2O))))

            testing_confusion_matrix[int(testing_examples[i][0])][int(np.argmax(output_layer_activations))] += 1

        testing_accuracy[epoch] = ((float(sum(testing_confusion_matrix.diagonal())) / 10000.0) * 100.0)
        print "Epoch ", epoch, ": ", "Testing Accuracy: ", testing_accuracy[epoch], "%"


    # Plot Graph
    plot_title = "No. of Hidden Units: "+ str(noofHiddenUnits)
    image_name = "HiddenUnits_"+str(noofHiddenUnits)+".jpg"
    plot.plot(training_accuracy)
    plot.plot(testing_accuracy)
    plot.ylabel('Accuracy%')
    plot.xlabel('Epoch')
    plot.title(plot_title)
    plot.savefig(image_name)

    np.set_printoptions(threshold=np.nan)
    print testing_confusion_matrix

main()