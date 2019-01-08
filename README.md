# Layers-Neurons
information about Neurons



## How to Create a Simple Neural Network in Python

Basically, an ANN comprises of the following components:

- An input layer that receives data and pass it on
- A hidden layer
- An output layer
- Weights between the layers
- A deliberate activation function for every hidden layer. In this simple neural network Python tutorial, we’ll employ the Sigmoid activation function.

There are several types of neural networks. In this project, we are going to create the feed-forward or perception neural networks. This type of ANN relays data directly from the front to the back.

Training the feed-forward neurons often need back-propagation, which provides the network with corresponding set of inputs and outputs. When the input data is transmitted into the neuron, it is processed, and an output is generated.

Here is a diagram that shows the structure of a simple neural network:

![redes 1](https://user-images.githubusercontent.com/17385297/50855199-f4ba8580-1365-11e9-9016-b9c7a00f2f4e.PNG)


And, the best way to understand how neural networks work is to learn how to build one from scratch (without using any library).

In this article, we’ll demonstrate how to use the Python programming language to create a simple neural network.

## The problem

Here is a table that shows the problem.

![redes 2](https://user-images.githubusercontent.com/17385297/50855205-fbe19380-1365-11e9-8c56-9456d994b2bf.PNG)


We are going to train the neural network such that it can predict the correct output value when provided with a new set of data.

As you can see on the table, the value of the output is always equal to the first value in the input section. Therefore, we expect the value of the output (?) to be 1.

Let’s see if we can use some Python code to give the same result (You can peruse the code for this project at the end of this article before continuing with the reading).

## Creating a NeuralNetwork Class

We’ll create a NeuralNetwork class in Python to train the neuron to give an accurate prediction. The class will also have other helper functions.

Even though we’ll not use a neural network library for this simple neural network example, we’ll import the numpy library to assist with the calculations.

The library comes with the following four important methods:
1. exp—for generating the natural exponential
2. array—for generating a matrix
3. dot—for multiplying matrices
4. random—for generating random numbers. Note that we’ll seed the random numbers to ensure their efficient distribution.

## Applying the Sigmoid function

We’ll use the Sigmoid function, which draws a characteristic “S”-shaped curve, as an activation function to the neural network.





This function can map any value to a value from 0 to 1. It will assist us to normalize the weighted sum of the inputs.

Thereafter, we’ll create the derivative of the Sigmoid function to help in computing the essential adjustments to the weights.

The output of a Sigmoid function can be employed to generate its derivative. For example, if the output variable is “x”, then its derivative will be x * (1-x).

## Training the model

This is the stage where we’ll teach the neural network to make an accurate prediction. Every input will have a weight—either positive or negative.

This implies that an input having a big number of positive weight or a big number of negative weight will influence the resulting output more.

Remember that we initially began by allocating every weight to a random number.

Here is the procedure for the training process we used in this neural network example problem:

1. We took the inputs from the training dataset, performed some adjustments based on their weights, and siphoned them via a method that computed the output of the ANN.
2. We computed the back-propagated error rate. In this case, it is the difference between neuron’s predicted output and the expected output of the training dataset.
3. Based on the extent of the error got, we performed some minor weight adjustments using the Error Weighted Derivative formula.
4. We iterated this process an arbitrary number of 15,000 times. In every iteration, the whole training set is processed simultaneously.

We used the “.T” function for transposing the matrix from horizontal position to vertical position. Therefore, the numbers will be stored this way:

![redes 3](https://user-images.githubusercontent.com/17385297/50855653-2f70ed80-1367-11e9-8669-474d311851c6.PNG)


Ultimately, the weights of the neuron will be optimized for the provided training data. Consequently, if the neuron is made to think about a new situation, which is the same as the previous one, it could make an accurate prediction. This is how back-propagation takes place.















artificial neural networks (ANNs)

Every network has a single input and output layers. The number of neurons in the input layer equals the number of input variables in the data being processed. The number of neurons in the output layer equals the number of outputs associated with each input. But the challenge is knowing the number of hidden layers and their neurons.


Here are some guidelines to know the number of hidden layers and neurons per each hidden layer in a classification problem:

- Based on the data, draw an expected decision boundary to separate the classes.
- Express the decision boundary as a set of lines. Note that the combination of such lines must yield to the decision boundary.
- The number of selected lines represents the number of hidden neurons in the first hidden layer.
- To connect the lines created by the previous layer, a new hidden layer is added. Note that a new hidden layer is added each time you need to create connections among the lines in the previous hidden layer.
- The number of hidden neurons in each new hidden layer equals the number of connections to be made. 

To make things clearer, let’s apply the previous guidelines for a number of examples.

> In artificial neural networks, hidden layers are required if and only if the data must be separated non-linearly.

Knowing that we need hidden layers to make us need to answer two important questions. These questions are:

1. What is the required number of hidden layers?
2. What is the number of the hidden neurons across each hidden layer? 
